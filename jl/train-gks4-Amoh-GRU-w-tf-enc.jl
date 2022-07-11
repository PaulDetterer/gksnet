## The Evaliable Commands
# 


using Glob
using Random
using WAV
using DSP
using BSON

using Flux
using CUDA
CUDA.device!(1)
CUDA.allowscalar(false)

## Load Prepared Data

dataset_path = "/mnt/space/datasets/googlespeech_v002"
all_commands = [c for c=readdir(dataset_path) if isdir(dataset_path*"/"*c)]

commands = [
 #   "_background_noise_",
    "down",
    "left",
    "right",
    "stop"
]
@assert all([c in all_commands for c=commands]) 
##


BSON.@load "train_ds_x.bson" train_ds_x
BSON.@load "train_ds_y.bson" train_ds_y
BSON.@load "val_ds_x.bson" val_ds_x
BSON.@load "val_ds_y.bson" val_ds_y
BSON.@load "test_ds_x.bson" test_ds_x
BSON.@load "test_ds_y.bson" test_ds_y

#Preprocess x
using Plots
@show size(train_ds_x)
@show size(val_ds_x)
function preprocess_x(x)
    x_new = Flux.upsample_bilinear(x,size=(32,32))
    # x_new = x
    x_new[:,:,1,:]
end

train_ds_x = preprocess_x(train_ds_x)
val_ds_x = preprocess_x(val_ds_x)
test_ds_x = preprocess_x(test_ds_x)
@show size(train_ds_x)
@show size(val_ds_x)
## Finally use the DataLoader

dev = gpu
trainingData = Flux.DataLoader((dev(train_ds_x), dev(train_ds_y)), batchsize=64,shuffle=true)
valData = Flux.DataLoader((dev(val_ds_x), dev(val_ds_y)), batchsize=64);

## Compute mean 

## Model Definition

model = Chain(
    GRU(32=>30),
    GRU(30=>20),
    Dense(20=>16,relu),
    Dense(16=>length(commands),σ),
    Dropout(0.2),
) |> dev
function resetmodel!(m)
    Flux.reset!(m[1])
    Flux.reset!(m[2])
end
model
##
# Checking 
resetmodel!(model)
# train_ds_x[1:1,:,1:1,1:2] |> display
model(dev(train_ds_x[1,:,1:2]))

## Loss Function
# pre = 4
function loss(x,y)
    resetmodel!(model)
    # Run for couple of rounds
   ŷ = sum([
        model(x[i,:,:])
        for i=(1:size(x)[1])
    ])
    Flux.Losses.logitcrossentropy(ŷ,y)
end
##
# Check Accuracy
function getAccuracy(m,d)
    acc = dev(0)
    for (x,y)=d
        resetmodel!(m)
        ŷ = sum([
            m(x[i,:,:,:]) 
            for i=1:size(x)[1]
        ])
        acc += sum(Flux.onecold(ŷ) .== Flux.onecold(y)) / size(x)[end]
    end
    acc/length(d)
end

# Total Loss
function loss_tot(d)
    l = dev(0)
    for (x,y) = d
        l+= loss(x,y)
    end
    return l/length(d)
end

## Goes very slow :''(
@show loss_tot(dev(valData)) 
@show getAccuracy(model,dev(valData))
## Train
# opt = Flux.Optimise.ADAM(4e-5)
opt = Flux.Optimise.ADAM()
ps = Flux.params(model)
loss_record = []
acc_record = []
#evalcb() = println("Loss: $(loss_tot(valData))")
for epoch=1:10
    testmode!(model,false)
    Flux.train!(
        loss,
        ps,
        trainingData,
        opt,
     #   cb=Flux.throttle(evalcb,10),
    )
    testmode!(model,true)
    l = loss_tot(valData)
    acc = getAccuracy(model,valData)
    println("E$(epoch): L: $l , Acc: $acc")
    push!(loss_record,l)
    push!(acc_record,acc)
end

## Plot Loss and Accuracy
fig1 = plot(
    loss_record,
    title = "Val Loss",
    legend=false,
    # xlabel="Epochs",
)
fig2 = plot(
    round.(acc_record.*100),
    title = "Val Accuracy",
    xlabel="Epochs",
    legend=false,
)
plot(fig1, fig2, layout=(2,1))
##
model = cpu(model)
##
resetmodel!(model)
ŷ = Flux.onecold(sum([model(test_ds_x[i,:,:]) for i=1:64]))
y = Flux.onecold(test_ds_y)
test_acc = sum(y .== ŷ) ./ length(y)
println("Test Accuracy: $(test_acc * 100)")


## Plot Confusion Matrix

conf_mat = [
    sum((y .== i) .& (ŷ .== j))
    for i=1:4, j=1:4
]
display(conf_mat)

heatmap(
    conf_mat,
    yflip = true,
    yticks = (1:8,commands),
    xticks = (1:8,commands),
)
## Save Model
using BSON:@save
@save "../models/GRU-model-ACC0p95.bson" model
## Free GPU from data set
trainingData = nothing
valData = nothing
GC.gc(true)
CUDA.reclaim()
##

