module MarkovChain
include("ReturnTimeEntropyEva.jl")
include("EntropyRateEva.jl")
include("HittingTimeEva.jl")
include("KemenyEva.jl")
include("Markov_or_not.jl")
include("Irreducible_or_not.jl")
include("MixingTimeEva.jl")
include("stadis.jl")
include("MC_COMP.jl")
include("ReturnTimeEntropyOp.jl")
include("EntropyRateOp.jl")
include("HittingTimeOp.jl")
include("KemenyOp.jl")
include("MixingTimeOp.jl")
include("MC_OP.jl")
# using IJulia
# IJulia.installkernel("Julia nodeps", "--depwarn=no")
using Reexport,Base,SparseArrays,LinearAlgebra,Ipopt,JuMP,Convex,SCS
@reexport using .ReturnTimeEntropyEvaMod,.HittingTimeEvaMod,.KemenyEvaMod,.MixingTimeEvaMod,.stadisMod,.EntropyRateEvaMod
@reexport using .ReturnTimeEntropyOpMod,.HittingTimeOpMod,.KemenyOpMod,.MixingTimeOpMod,.EntropyRateOpMod
@reexport using .MC_COMP_MOD,.MC_OP_MOD
export Markov_or_not
export Irreducible_or_not


end # module
