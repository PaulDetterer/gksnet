## The Evaliable Commands
# 
dataset_path = "/mnt/space/datasets/mixed_speech_commands"
commands = [c for c=readdir(dataset_path) if c!="README.md"]

##


# Read out files 

# Gather file pathes
using Glob
using Random

filenames = glob("$(relpath(dataset_path))/*/*") |> shuffle
num_examples = length(readdir("$(relpath(dataset_path))/$(commands[1])"))
println("Number of total examples: ",length(filenames))
println("Number of examples per label: ", num_examples)
println("Example File Tensor: ", filenames[1])

# Separate the data set in training data, validation data and test data
train_num = div(length(filenames) * 80, 100)
val_num = div(length(filenames) - train_num, 2)
test_num = length(filenames) - train_num - val_num
@show (train_num, val_num, test_num)
train_files = filenames[1:train_num]
val_files = filenames[(train_num+1):(train_num+val_num)]
test_files = filenames[(train_num+val_num+1):end]

println("Training set size: ", length(train_files))
println("Validation set size: ", length(val_files))
println("Test set size: ", length(test_files))
@show length(filenames)

##
# Check loading WAVs
using WAV
test_audio, fs = wavread("$(dataset_path)/down/0a9f9af7_nohash_0.wav")
@show fs
@show size(test_audio)
#inline_audioplayer(test_audio,fs)
##wavplay(test_audio,fs)
##

# Function to load WAVs
function decode_audio(audio_binary)
    audio, fs = wavread(audio_binary)
    convert.(Float32,audio[:,1])
end
decode_audio("$(dataset_path)/down/0a9f9af7_nohash_0.wav")
##

# Function to get command label
function get_label(file_path)
    split(file_path,"/")[end-1] # For Windows System otherwise replace with /
end
get_label(filenames[1])
##
# Load waveforms
train_wf_x = decode_audio.(train_files)
train_wf_y = get_label.(train_files)
val_wf_x = decode_audio.(val_files)
val_wf_y = get_label.(val_files)
test_wf_x = decode_audio.(test_files)
test_wf_y = get_label.(test_files)
##
using Plots

rows = 3
cols = 3

n = rows*cols

figs = [
    plot(
        (1:length(train_wf_x[i]))./1e3, train_wf_x[i],
        title = train_wf_y[i],
        label = "Julia",
        yticks = -1.2:0.4:1.2,
        ylim = (-1.3,1.3),
        xticks = 0:5:15,
        legend=true,
    )
    for i=1:n
]


plot(
    figs...,
    layout=(rows,cols),
    size=(600,800),
)
##

# Compute spectragrams (with stft)
# Compared to tf function the spectrogram from DSP.stft is transposed

using DSP

function get_spectrogram(waveform)
    input_len = 16000
    waveform = length(waveform) >= input_len ? 
                    waveform[1:input_len]    :
                    [convert.(Float32,waveform);zeros(Float32,input_len - length(waveform))]
    spectrogram = DSP.stft(
        waveform,
        255,
        255-128, 
        window=hanning,
        #fs=fs,
    ) |> x->abs.(x)
    spectrogram'
end

get_spectrogram(train_wf_x[1])
##
# Function to plot spectrograms
# I was too lazy to fix the xticks of the spectrogram

function plot_spectrogram(spectrogram, title)
    tmp = similar(spectrogram)
    if (length(size(spectrogram))>2)
        @assert length(size(spectrogram)) == 3
        tmp = spectrogram[:,:,1]
    else
        tmp = spectrogram
    end
    log_spec = log.(tmp .+ eps(Float32))
    heatmap(log_spec', legend=false, title=title)
end


# tst_waveform = decode_audio("../dataset/mini_speech_commands/down/0a9f9af7_nohash_0.wav")
# tst_spectrogram = get_spectrogram(tst_waveform)
tst_spectrogram= get_spectrogram(train_wf_x[1])

p1 = plot(train_wf_x[1])
p2 = plot_spectrogram(tst_spectrogram, "test spectrogram")

plot(p1,p2, layout=(2,1))
##
using Flux
using CUDA
CUDA.allowscalar(false)

# Function to get spectrogram label pairs
# Deviation from TF - I store labels in onehot format
function my_spectrogram(waveform)
    spectrogram = convert.(Float32,get_spectrogram(waveform))
    reshape(spectrogram,size(spectrogram)...,1)
end

get_label_id(label) = Flux.onehot(label,commands)


# Check the function
#x,y = get_spectrogram(train_wf_x[1]), get_label_id(train_wf_y[1])
x = my_spectrogram(train_wf_x[1])
y = get_label_id(train_wf_y[1])

p1 = plot(train_wf_x[1],title="$(train_wf_y[1])")
@show size(x)
p2 = plot_spectrogram(x, "Spectogram")
@show commands
plot(p1, p2, layout=(2,1)) 
##

spectrogram_ds = get_spectrogram.(train_wf_x)
spectrogram_id = get_label_id.(train_wf_y)

spectrogram_ds[1]

rows = 3
cols = 3
n = rows*cols

figs = [
    plot_spectrogram(spectrogram_ds[i][:,:,1,1],"$i : $(first(commands[spectrogram_id[i]]))")
    for i=1:n
]

plot(figs...,layout=(rows,cols))
##

# Function to prepare all data needed for training, validation and testing
function preprocess_dataset(files)
    waveform_ds = decode_audio.(files) 
    labels = get_label.(files)
    x = get_spectrogram.(waveform_ds)
    y = get_label_id.(labels)
    if (length(x)<=3200)
        x = cat(x...,dims=4)
    else
        xt = cat(x[1:3200]...,dims=4)
        l = length(x)
        for i=2:div(l,3200)
            xt = [xt;;;;cat(x[(3200*i-3199):3200*i]...,dims=4)]
        end
        if (l % 3200 > 0)
            x = [xt;;;; cat(x[(l-(l % 3200)+1):end]...,dims=4)]
        else
            x = xt
        end
    end
    x, cat(y...,dims=2)
end

function preprocess_dataset_from_tf(path, num)
    wf_x, wf_y = load_tf_waveform_dump(path, num)
    x = my_spectrogram.(wf_x)
    y = get_label_id.(wf_y)
    if (length(x)<=3200)
        x = cat(x...,dims=4)
    else
        x1 = cat(x[1:3200]...,dims=4)
        x2 = cat(x[3201:end]...,dims=4)
        x = [x1;;;;x2;;;;]
    end
    x , cat(y...,dims=2)
end

#val_ds_x, val_ds_y = preprocess_dataset_from_tf("../dataset/TF/val", 800)
val_ds_x, val_ds_y = preprocess_dataset(val_files)
val_ds_x

## Prepare Data
train_ds_x, train_ds_y = preprocess_dataset(train_files)
val_ds_x, val_ds_y = preprocess_dataset(val_files)
test_ds_x, test_ds_y = preprocess_dataset(test_files)
# train_ds_x, train_ds_y = preprocess_dataset_from_tf("../dataset/TF/train",6400)
# val_ds_x, val_ds_y = preprocess_dataset_from_tf("../dataset/TF/val", 800)
# test_ds_x, test_ds_y = preprocess_dataset_from_tf("../dataset/TF/test", 800);

## Check data
@show size(train_ds_x)
@show size(train_ds_y)

rows = 3
cols = 3
n = rows*cols

figs = [
    plot_spectrogram(train_ds_x[:,:,1,i],"$i : $(first(commands[train_ds_y[:,i]]))")
    for i=1:n
]

plot(figs...,layout=(rows,cols))

## Finally use the DataLoader

trainingData = Flux.DataLoader((gpu(train_ds_x), gpu(train_ds_y)), batchsize=64,shuffle=true)
valData = Flux.DataLoader((gpu(val_ds_x), gpu(val_ds_y)), batchsize=64)

## Compute mean 
using Statistics
ds_mean =  mean(train_ds_x)
ds_sqrt_var = sqrt(var(train_ds_x))

## Model Definition

model = Chain(
    Upsample(:bilinear,size=(32,32)),
    x->(x .- ds_mean) ./ ds_sqrt_var,
    Conv((3,3),1=>32, Flux.relu),
    Conv((3,3),32=>64, Flux.relu),
    MaxPool((2,2)),
    Dropout(0.25),
    Flux.flatten,
    Dense(12544=>128,relu),
    Dropout(0.5),
    Dense(128=>length(commands)),
) |> gpu
## Checking 
model(gpu(train_ds_x[:,:,1:1,1:2]))

## Loss Function
function loss(x,y)
    ŷ = model(x)
    Flux.Losses.logitcrossentropy(ŷ,y)
end

# Check Accuracy
function getAccuracy(m,d)
    acc = gpu(0)
    for (x,y)=d
        ŷ = model(x)
        acc += sum(Flux.onecold(ŷ) .== Flux.onecold(y)) / size(x)[end]
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
@show loss_tot(gpu(valData)) 
@show getAccuracy(model,gpu(valData))
## Train
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
##
GC.gc(true)
CUDA.reclaim()
using BSON:@save
@save "../models/tf-inspired-model-ACC0p95.bson" model
##
