using Plots
import Distributions; dst = Distributions
import DataStructures

struct Lattice
    memory::DataStructures.OrderedDict{Tuple, Int8}
    neib::Tuple
end

function Lattice(lattice::Lattice, key::Tuple, value)
    lattice.memory[key] = value
end

function Lattice(lattice::Lattice, key::Tuple)
    if ~haskey(lattice.memory, key)
        lattice.memory[key] = 1
    end
    return lattice.memory[key]
end

function Lattice()
    return Lattice(DataStructures.OrderedDict{Tuple, Int8}((0,0,0)=>0), (
        ( 1, 0, 0), 
        ( 0, 1, 0), 
        ( 0, 0, 1), 
        (-1, 0, 0), 
        ( 0,-1, 0), 
        ( 0, 0,-1)))
end

function len(lattice::Lattice)
    return length(lattice.memory)
end

function Ω(x; a=1, b=1)
    """
    # Отображаем lattice_weigths в weigths используя параметры a и b
    0 => a
    2 => 1
    3:7 => b
    """
    if x == 0
        return a
    elseif x == 2
        return 1.0
    else
        return b
    end
end    

function InitSim()
    lattice = Lattice()
    pos = (0,0,0)
    Lattice(lattice, pos, 0)
    for neib in lattice.neib
        Lattice(lattice, pos .+ neib, Lattice(lattice, pos .+ neib) + 1)
    end
    lattice_weigths = [Lattice(lattice, pos .+ neib) for neib in lattice.neib]
    weigths = Ω.(lattice_weigths; a=aa, b=bb)
    probs = weigths ./ sum(weigths)
    return lattice, probs, pos
end

function Update!(lattice, probs, pos, aa, bb)
    step = lattice.neib[rand(dst.Categorical(probs))]
    if Lattice(lattice, pos .+ step) == 0
        pos = pos .+ step
        lattice_weigths = [Lattice(lattice, pos .+ neib) for neib in lattice.neib]
        weigths = Ω.(lattice_weigths; a=aa, b=bb)
        probs = weigths ./ sum(weigths)
    else
        pos = pos .+ step
        Lattice(lattice, pos, 0)
        for neib in lattice.neib
            if Lattice(lattice, pos .+ neib) != 0
                Lattice(lattice, pos .+ neib, Lattice(lattice, pos .+ neib) + 1)
            end
        end
        lattice_weigths = [Lattice(lattice, pos .+ neib) for neib in lattice.neib]
        weigths = Ω.(lattice_weigths; a=aa, b=bb)
        probs = weigths ./ sum(weigths)    
    end
    return probs, pos
end

log_param_a = rand()*8-4
log_param_b = rand()*8-4

aa = exp(log_param_a)
bb = exp(log_param_b)
lattice, probs, pos = InitSim()
len_lattice = len(lattice)
count = 0

time_array = []
start_time = time()
len_lattice = len(lattice)
while len(lattice) < 10000
    global probs
    global pos
    global len_lattice
    probs, pos = Update!(lattice, probs, pos, aa, bb)
    if len(lattice) > len_lattice 
        len_lattice = len(lattice)
        temp_time = time()
        if rand()<0.01
            println(len_lattice, "  --> ", (temp_time-start_time))
        end
        push!(time_array, temp_time-start_time)
    end 
end

using NPZ
res = hcat([[k for k in i] for i in keys(lattice.memory) if lattice.memory[i]==0]...)
npzwrite("data/a=$(log_param_a)_b=$(log_param_b)_visited_volume=10k.npy", res)