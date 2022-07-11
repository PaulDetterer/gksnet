## The Evaliable Commands
# 


using Glob
using Random
using WAV
using DSP

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

# Gather file pathes

filenames = vcat(
    [
        glob("$(relpath(dataset_path))/$c/*.wav")
        for c=commands
    ]...) |> shuffle
num_examples = length(readdir("$(relpath(dataset_path))/$(commands[1])"))
println("Number of total examples: ",length(filenames))
println("Number of examples per label: ", num_examples)
println("Example File Tensor: ", filenames[1])
# Separate data according to googles method
val_files = [
    "$(relpath(dataset_path))/$f"
    for f=open(f->split(strip(read(f,String)),"\n"),"$(dataset_path)/validation_list.txt")
        if split(f,"/")[1] in commands
]
test_files = [
    "$(relpath(dataset_path))/$f"
    for f=open(f->split(strip(read(f,String)),"\n"),"$(dataset_path)/testing_list.txt")
        if split(f,"/")[1] in commands
]
train_files = [f for f=filenames if (!(f in val_files) && !(f in test_files))] |> shuffle

println("Training set size: ", length(train_files))
println("Validation set size: ", length(val_files))
println("Test set size: ", length(test_files))
@show length(filenames)

# Function to load WAVs
function decode_audio(audio_binary)
    # println("Reading $(audio_binary)")
    audio, fs = wavread(audio_binary)
    convert.(Float32,audio[:,1])
end
# Function to get command label
function get_label(file_path)
    split(file_path,"/")[end-1] # For Windows System otherwise replace with /
end

# Load waveforms
train_wf_x = decode_audio.(train_files)
train_wf_y = get_label.(train_files)
val_wf_x = decode_audio.(val_files)
val_wf_y = get_label.(val_files)
test_wf_x = decode_audio.(test_files)
test_wf_y = get_label.(test_files)


# Compute Spectrogram
function get_spectrogram(waveform)
    spectrogram = DSP.stft(
        waveform[1:2:end],
        127,
        0, 
        window=hanning,
        nfft=127,
        #fs=fs,
    ) |> x->abs.(x)
    #if 
    len = size(spectrogram)[2]
    len >= 64 ?
        spectrogram[:,1:64]' :
        [spectrogram zeros(Float32,64,64-len)]'
end

get_label_id(label) = Flux.onehot(label,commands)
function preprocess_dataset(files)
    waveform_ds = decode_audio.(files)
    labels = get_label.(files)
    x = get_spectrogram.(waveform_ds)
    xt = x
    if (length(xt)<=3200)
        xt = cat(x...,dims=3)
    else
        xt = cat(x[1:3200]...,dims=3)
        for i=2:div(length(x),3200)
            xt = [xt;;;cat(x[(3200i-3199):(3200i)]...,dims=3)]
        end
        xt = [xt;;;cat(x[end-(length(x)%3200)+1:end]...,dims=3)]
    end
    y = get_label_id.(labels)
    xt, cat(y...,dims=2)
end

train_ds_x, train_ds_y = preprocess_dataset(train_files)
val_ds_x, val_ds_y = preprocess_dataset(val_files)
test_ds_x, test_ds_y = preprocess_dataset(test_files)

##
@show size(train_ds_x)
@show size(val_ds_x)
## Finally use the DataLoader

dev = cpu
trainingData = Flux.DataLoader((dev(train_ds_x), dev(train_ds_y)), batchsize=64,shuffle=true)
valData = Flux.DataLoader((dev(val_ds_x), dev(val_ds_y)), batchsize=64);

## Compute mean 

## Model Definition

model = Chain(
    GRU(64=>30),
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
            model(x[i,:,:,:]) 
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
opt = Flux.Optimise.ADAM(4e-5)
# opt = Flux.Optimise.ADAM()
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

