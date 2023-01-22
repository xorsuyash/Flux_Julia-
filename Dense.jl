

#building layers 

#=W1=rand(3,5)
b1=rand(3)

layer1(x)=W1*x.+b1

W2=rand(2,3)
b2=rand(2)
layer2(x)=W2*x.+b2
model(x)=layer2(σ.(layer1(x)))
model(rand(5))=#

#=function linear(in,out)
    W=rand(out,in)
    b=rand(out)
    x->W*x.+b
end 

linear1=linear(5,3)
linear2=linear(3,2)

model(x)=linear2(σ.(linear1(x)))
model(rand(5))=#

#building dense layer 
using Flux 
struct neural 
    W
    b
end 

neural(in::Integer,out::Integer)=neural(randn(out,in),randn(out))
(m::neural)(x)=m.W*x+m.b
layer1=neural(10,5)
layer2=neural(5,2)
model(x)=layer2(σ.(layer1(x)))
model(rand(10))


