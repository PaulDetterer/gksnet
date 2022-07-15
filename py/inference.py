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
    for ich in range(input.shape[2]):
        for och in range(weights.shape[3]):
            for ix in range(input.shape[1]-kx+1):
                for iy in range(input.shape[0]-ky+1):
                    output[iy,ix,och] = np.sum(
                        np.multiply(
                            weights[::-1,::-1,ich,och],
                            input[iy:iy+kx,ix:ix+ky,ich])) + biases[och]

    return output
def _bn_layer(input,mu,sigmasq,beta,gamma):
    output = copy(input)
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

##