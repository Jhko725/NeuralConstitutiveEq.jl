##
using Printf
using Statistics
s = 0
s = "Dog"
println(s)

function changeNum()
    x::Int8 = 10
    y::String = "Dog"
end

changeNum()

##
bF = 1.111111111111
println(bF + 0.111111111111)

##
c2 = Char(120)
println(c2)

##
i1 = UInt8(trunc(3.14))
println(i1)

f1 = parse(Float64, "1")
println(f1)

i2 = parse(Int8, "1")
println(i2)

##
s1 = "Just some random words\n"
println(length(s1))

println(s1[1])
println(s1[end])
print(s1[1:4])
s2 = string("Yukiteru","Amano")
println(s2)
println("Yuno","Gasai")
i3 = 2
i4 = 3
println("$i3 + $i4 = $(i3 + i4)")

##
s3 = """
have
many
lines"""

println("Takao">"Hiyama")
println(findfist(isequl('i'), "Keigo"))
