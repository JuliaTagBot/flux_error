const DIRS = [:left, :right, :up, :down]
const VALS = vcat(0,[(2).^(1:13)...])
using StatsBase, DataFrames, Flux, Plots, JLD2, FileIO, BSON, Random

function grid_to_feature(grid)
    vec(Flux.onehotbatch(vec(grid),VALS))
end

policy = Flux.Chain(
  grid_to_feature,
  Dense(16*14, 128, relu),
  Dense(128, 4, relu)
  ,softmax
  )

loss(x, (move, reward)) = -reward*Flux.crossentropy(policy(x), Flux.onehot(move, DIRS))
opt=ADAM(Flux.params(policy))

@load "res_vec.jld2" res_vec

Flux.train!(loss, res_vec, opt)
