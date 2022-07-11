## Load Tools


using Glob
using Random
using WAV
using DSP

using Flux
using CUDA
CUDA.device!(1)
CUDA.allowscalar(false)

## The Evaliable Commands
# 
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


## Load Prepared Data
using BSON: @load

@load "train_ds_x.bson" train_ds_x
@load "train_ds_y.bson" train_ds_y
@load "val_ds_x.bson" val_ds_x
@load "val_ds_y.bson" val_ds_y
@load "test_ds_x.bson" test_ds_x
@load "test_ds_y.bson" test_ds_y



## Finally use the DataLoader

trainingData = Flux.DataLoader((gpu(train_ds_x), gpu(train_ds_y)), batchsize=64,shuffle=true)
valData = Flux.DataLoader((gpu(val_ds_x), gpu(val_ds_y)), batchsize=64);

## Compute mean 
using Statistics
ds_mean =  mean(train_ds_x)
ds_sqrt_var = sqrt(var(train_ds_x))

## Model Definition

model = Chain(
    # x->[x;x], # Needed for upsample to work
    # Upsample(:bilinear,size=(2,32)),
    # x->(x .- ds_mean) ./ ds_sqrt_var,
    # x->x[1:1,:,:,:], # Get back to right dimensions 
    # Conv((1,3),1=>32, Flux.relu),
    # Conv((1,3),32=>64, Flux.relu),
    # MaxPool((1,2)),
    # Dropout(0.25),
    Flux.flatten,    
    RNN(129=>256),
    Dense(256=>length(commands)),
    Dropout(0.2),
) |> gpu

function resetmodel!(m)
    Flux.reset!(m[2])
end
# Checking 
dev = gpu
resetmodel!(model)
# train_ds_x[1:1,:,1:1,1:2] |> display
model(dev(train_ds_x[1:1,:,:,1:2]))

## Loss Function
function loss(x,y)
    resetmodel!(model)
    # Run for couple of rounds
    ŷ = sum([
        model(x[i:i,:,:,:])
        for i=1:size(x)[1]
    ])
    Flux.Losses.logitcrossentropy(ŷ,y)
end
##
# Check Accuracy
function getAccuracy(m,d)
    acc = gpu(0)
    for (x,y)=d
        resetmodel!(m)
        ŷ = sum([m(x[i:i,:,:,:]) for i=1:size(x)[1]])
        acc += sum(Flux.onecold(ŷ) .== Flux.onecold(y)) /size(x)[end]
    end
    acc/length(d)
end

# Total Loss
function loss_tot(d)
    l = gpu(0)
    for (x,y) = d
        l+= loss(x,y)
    end
    return l/length(d)
end

## Goes very slow :''(
resetmodel!(model)
@show loss_tot(dev(valData)) 
@show getAccuracy(model,dev(valData))
## Train
# opt = Flux.Optimise.ADAM(1e-5)
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
ŷ = Flux.onecold(model(test_ds_x))
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
@save "../models/tf-inspired-model-ACC0p95.bson" model
## Free GPU from data set
trainingData = nothing
valData = nothing
GC.gc(true)
CUDA.reclaim()
##

