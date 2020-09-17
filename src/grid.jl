
abstract type AbstractCoord{N,T} end

coord_axis(A::AbstractCoord{N,T}) where{N,T} = N
coord_eltype(A::AbstractCoord{N,T}) where{N,T} = T

struct CartesianCoord{N,T<:Real} <: AbstractCoord{N,T}
    name  :: String
    min   :: T
    max   :: T
    nodes :: Int
end

struct GaussLobattoCoord{N,T<:Real} <: AbstractCoord{N,T}
    name  :: String
    min   :: T
    max   :: T
    nodes :: Int
end

coord_type(coord::CartesianCoord)    = "Cartesian"
coord_type(coord::GaussLobattoCoord) = "GaussLobatto"

struct Cartesian{N} end
function Cartesian{N}(name::String, xmin::T, xmax::T, nodes::Integer;
                      endpoint::Bool=true) where {T<:Real,N}
    min_ = xmin
    if (!endpoint)
        h    = (xmax - xmin) / nodes
        max_ = xmax - h
    else
        max_ = xmax
    end
    CartesianCoord{N,T}(name, min_, max_, nodes)
end
Cartesian(args...; endpoint=true) = Cartesian{1}(args...; endpoint=endpoint)

struct GaussLobatto{N} end
function GaussLobatto{N}(name::String, xmin::T, xmax::T,
                          nodes::Integer) where {T<:Real,N}
    GaussLobattoCoord{N,T}(name, xmin, xmax, nodes)
end
GaussLobatto(args...) = GaussLobatto{1}(args...)

struct Coord{N} end
function Coord{N}(coord_type::String, name::String, min::T, max::T, nodes::Int) where {T<:Real,N}
    if coord_type == "Cartesian"
        return CartesianCoord{N,T}(name, min, max, nodes)
    elseif coord_type == "GaussLobatto"
        return GaussLobattoCoord{N,T}(name, min, max, nodes)
    else
        error("Unknown coord type")
    end
end


@inline function delta(coord::CartesianCoord) where {T<:Real,N}
    (coord.max - coord.min) / (coord.nodes - 1)
end

# a Gauss-Lobatto grid has non-uniform spacing. not sure if the best is to
# return "missing", not define the method, or just have it return NaN, as now.
@inline delta(coord::GaussLobattoCoord) = NaN

@inline function Base.getindex(coord::CartesianCoord, i::Int)
    h = delta(coord)
    coord.min + (i - 1) * h
end

@inline function Base.getindex(coord::GaussLobattoCoord, j::Int)
    xmin  = coord.min
    xmax  = coord.max
    M     = coord.nodes - 1.0
    xj    = -cos( (j - 1.0) * pi / M)
    0.5 * (xmax + xmin + (xmax - xmin) * xj)
end

@inline Base.firstindex(coord::AbstractCoord) = 1
@inline Base.lastindex(coord::AbstractCoord)  = coord.nodes
@inline Base.length(coord::AbstractCoord) = coord.nodes

@inline Base.getindex(coord::AbstractCoord, ::Colon) = [coord[i] for i in 1:coord.nodes]


struct Chart{N,A}
    coords :: A
    function Chart{A}(coords) where {A}
        N = length(coords)
        for a in 1:N
            @assert(coord_axis(coords[a]) == a, "wrong order in grid array")
        end
        new{N,A}(coords)
    end
end
"""
    Chart(coords::A)

A `Chart` is a collection of `AbstractCoord`s
"""
Chart(coords::A) where {A} = Chart{A}(coords)

Chart(coords::Vararg{AbstractCoord,N}) where {N} = Chart(coords)

function Chart(coord_types::Vector, names::Vector, mins::Vector, maxs::Vector,
               nodess)
    dim_ = length(names)
    @assert(length(mins) == length(maxs) == length(nodess) == length(coord_types) == dim_)
    coords_ = [Coord{i}(coord_types[i], names[i], mins[i], maxs[i], nodess[i]) for i in 1:dim_]
    coords  = Tuple(coords_)
    Chart(coords)
end


@inline Base.ndims(chart::Chart{N}) where {N} = N

@inline function Base.getindex(chart::Chart{N}, idx::Vararg{Int,N}) where {N}
    [chart.coords[a][idx[a]] for a in 1:N]
end

@inline function Base.getindex(chart::Chart{N}, ::Colon) where {N}
    [chart.coords[a][:] for a in 1:N]
end

@inline function name(chart::Chart{N}) where {N}
    [chart.coords[a].name for a in 1:N]
end

@inline function min(chart::Chart{N}) where {N}
    [chart.coords[a].min for a in 1:N]
end

@inline function max(chart::Chart{N}) where {N}
    [chart.coords[a].max for a in 1:N]
end

@inline function nodes(chart::Chart{N}) where {N}
    [chart.coords[a].nodes for a in 1:N]
end

@inline function delta(chart::Chart{N}) where {N}
    [delta(chart.coords[a]) for a in 1:N]
end

@inline function coord_type(chart::Chart{N}) where {N}
    [coord_type(chart.coords[a]) for a in 1:N]
end

@inline Base.size(chart::Chart) = Tuple(nodes(chart))


struct Atlas{N,A<:Chart}
    charts :: NTuple{N,A}
end

"""
    Atlas(charts::A)

An `Atlas` is a collection of `Chart`s
"""
Atlas(charts::Vararg{Chart,N}) where {N} = Atlas(charts)
Atlas(xx::Vector) = Atlas(Tuple(xx))

function Atlas(chart::Chart)
    charts = (chart)
    Atlas(charts)
end
