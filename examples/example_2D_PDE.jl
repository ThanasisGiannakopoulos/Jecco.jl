
using Jecco
using LinearAlgebra
using SparseArrays
using Plots

source(x,y) = exp(-x^2 - y^2) * (-4 + 3 * (x^2 + y^2) + 4 * x^2 * y^2)

# returns axx Dxx + ayy Dyy + axy Dxy + bx Dx + by Dy + cc. note that this
# function overwrites the input matrices to save memory
function build_operator(Dxx::SparseMatrixCSC, Dyy::SparseMatrixCSC, Dxy::SparseMatrixCSC,
                        Dx::SparseMatrixCSC, Dy::SparseMatrixCSC,
                        axx::Vector, ayy::Vector, axy::Vector,
                        bx::Vector, by::Vector, cc::Vector)
    Jecco.mul_col!(axx, Dxx)
    Jecco.mul_col!(ayy, Dyy)
    Jecco.mul_col!(axy, Dxy)
    Jecco.mul_col!(bx,  Dx)
    Jecco.mul_col!(by,  Dy)
    ccId = Diagonal(cc)

    Dxx + Dyy + Dxy + Dx + Dy + ccId
end

#=
use the Kronecker product (kron) to build the 2-dimensional derivation matrices
from the 1-dimensional ones. see for instance:

  https://en.wikipedia.org/wiki/Kronecker_product

  https://arxiv.org/pdf/1801.01483.pdf (section 5)
=#
function deriv_operators(hx, hy, Nx::Int, Ny::Int, ord::Int)
    Dx_op  = CenteredDiff{1}(1, ord, hx, Nx)
    Dxx_op = CenteredDiff{1}(2, ord, hx, Nx)

    Dy_op  = CenteredDiff{2}(1, ord, hy, Ny)
    Dyy_op = CenteredDiff{2}(2, ord, hy, Ny)

    Dx  = kron(I(Ny), SparseMatrixCSC(Dx_op))
    Dxx = kron(I(Ny), SparseMatrixCSC(Dxx_op))
    Dy  = kron(SparseMatrixCSC(Dy_op), I(Nx))
    Dyy = kron(SparseMatrixCSC(Dyy_op), I(Nx))

    Dx, Dy, Dxx, Dyy, Dx * Dy
end


x_min    = -5.0
x_max    =  5.0
x_nodes  =  128
y_min    = -5.0
y_max    =  5.0
y_nodes  =  64

ord = 4

xcoord  = Cartesian{1}("x", x_min, x_max, x_nodes, endpoint=false)
ycoord  = Cartesian{2}("y", y_min, y_max, y_nodes, endpoint=false)

hx = Jecco.delta(xcoord)
hy = Jecco.delta(ycoord)

Nx = xcoord.nodes
Ny = ycoord.nodes

f_exact = [exp(-xcoord[i]^2 - ycoord[j]^2) for i in 1:Nx, j in 1:Ny]


Dx, Dy, Dxx, Dyy, Dxy = deriv_operators(hx, hy, Nx, Ny, ord)

f0    = zeros(Nx,Ny)
ind2D = LinearIndices(f0)

M = Nx * Ny

b_vec = zeros(M)

axx     = ones(M)
ayy     = ones(M)
axy     = zeros(M)
bx      = zeros(M)
by      = zeros(M)
cc      = zeros(M)

for j in 1:Ny, i in 1:Nx
    idx = ind2D[i,j]

    xi  = xcoord[i]
    yi  = ycoord[j]

    axy[idx] = xi * yi
    bx[idx]  = xi
    by[idx]  = yi
    cc[idx]  = xi^2 + yi^2

    b_vec[idx] = source(xi, yi)
end

# build operator A = Dxx + Dyy + x y Dxy + x Dx + y Dy + (x^2 + y^2)
A_mat = build_operator(Dxx, Dyy, Dxy, Dx, Dy, axx, ayy, axy, bx, by, cc)

# since we're using periodic boundary conditions, the operator A_mat (just like
# the Dx and Dxx operators) is strictly speaking not invertible (it has zero
# determinant) since the solution is not unique. indeed, its LU decomposition
# shouldn't even be defined. for some reason, however, the call to "lu" does in
# fact factorize the matrix. in any case, to be safer, let's instead call
# "factorize", which uses fancy algorithms to determine which is the best way to
# factorize (and which performs a QR decomposition if the LU fails). the inverse
# that is performed probably returns the minimum norm least squares solution, or
# something similar. in any case, for our purposes here we mostly care about
# getting a solution (not necessarily the minimum norm least squares one).
A_fact = factorize(A_mat)
sol    = A_fact \ b_vec

@inbounds for idx in eachindex(f0)
    f0[idx] = sol[idx]
end

j_slice = div(Ny,2) + 1
x = xcoord[:]

plot(x, f_exact[:,j_slice])
scatter!(x, f0[:,j_slice])
