##
import numpy as np

def _relu(x):
    return np.maximum(0,x)

def _conv_layer(input,weights,biases):
    output = np.zeros(
        (
            input.shape[0]-weights.shape[0]+1,
            input.shape[1]-weights.shape[1]+1,
            weights.shape[3]
        ),
        dtype=np.float32
    )
    ky = weights.shape[0]
    kx = weights.shape[1]
    Ich = weights.shape[2]
    # print("Layer Dimensions: ", kx, ky, Ich)
    # print("Input Shape: ", input.shape)
    for och in range(weights.shape[3]):
        for ix in range(input.shape[1]-kx+1):
            for iy in range(input.shape[0]-ky+1):
                for ich in range(Ich):
                    # print("Output Shape: ", output[iy, ix, och].shape)
                    output[iy,ix,och] += np.sum(
                        np.multiply(
                            weights[::-1,::-1,ich,och],
                            input[iy:iy+kx,ix:ix+ky,ich]))
                output[iy,ix,och] += biases[och]

    return output
def _bn_layer(input,mu,sigmasq,beta,gamma):
    output = input.copy()
    for ch in range(input.shape[2]):
        output[:,:,ch] = (((input[:,:,ch] - mu[ch])/(np.sqrt(sigmasq[ch]+1e-5)))*gamma[ch])+beta[ch]
    return output

def _maxpool_layer(input):
    output = np.zeros((input.shape[0]//2,input.shape[1]//2,input.shape[2]),dtype=np.float32)
    for y in range(input.shape[0]//2):
        for x in range(input.shape[1]//2):
            for ch in range(input.shape[2]):
                output[y,x,ch] = np.max(input[(y*2):(y*2+2),(x*2):(x*2+2),ch])
    return output

def _cnn_infer(input,params):
    #1 - Conv
    output = _conv_layer(input,params[1]["weight"],params[1]["bias"]) #1
    output = _relu(output) #1
    #2 - BatchNormdfaf
    output = _bn_layer(
        output,
        params[2]["mu"],
        params[2]["sigmasq"],
        params[2]["beta"],
        params[2]["gamma"]
    )
    # #3 - Conv
    output = _conv_layer(output,params[3]["weight"],params[3]["bias"])
    output = _relu(output) 
    # #4 - BatchNorm
    output = _bn_layer(
        output,
        params[4]["mu"],
        params[4]["sigmasq"],
        params[4]["beta"],
        params[4]["gamma"]
    )
    # #5 - Maxpool
    output = _maxpool_layer(output)
    #6 - Dense
    #7 - Dense
    return output


##