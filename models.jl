using Flux
predict(x)=W*x.+b
s=0
function loss(x,y)
    y_hat=predict(x)
    loss=sum((y_hat.-y).^2)
end 
x=rand(5)
y=rand(2)
W=rand(2,5)
b=rand(2)
loss1=loss(x,y)
gs=gradient(()->loss(x,y),Flux.params(W,b))
w_hat=gs[W]
W.-=0.1.*w_hat
loss2=loss(x,y)
println("first loss is $(loss1) and second loss is $(loss2)")


