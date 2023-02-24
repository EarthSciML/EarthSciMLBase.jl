export partialderivatives_lonlat2xymeters

# Convert degrees to radians
deg2rad(x) = x * π / 180.0
# Scaling factor for converting latitude to meters
const lat2meters = 111.32e3
# Return scaling factor for converting longitude to meters at given latitude
lon2meters(lat) = 40075.0e3 * cos(deg2rad(lat)) / 360.0
# return index in vector of partial-independent variables of variable with given name
varindex(pvars::AbstractVector, varname::Symbol) = findfirst(nameof.(pvars) .== varname)

"""
$(SIGNATURES)

Return the partial derivative operators corresponding to each of the given partial-independent variables.
"""
function partialderivatives_identity(pvars::AbstractVector)
    Differential.(pvars)
end

"""
$(SIGNATURES)

Return the partial derivative operators corresponding to each of the given partial-independent variables
after converting variables named `lon` and `lat` from degrees to x and y meters, 
assuming they represent longitude and latitude on a spherical Earth.
"""
function partialderivatives_lonlat2xymeters(pvars::AbstractVector; default_lat=0.0)
    latindex = varindex(pvars, :lat)
    lonindex = varindex(pvars, :lon)
    if !isnothing(latindex)
        lat = pvars[latindex]
    else
        lat = default_lat
    end


    δs = Differential.(pvars)

    Dx(var) = δs[lonindex](var) / lon2meters(lat)
    Dy(var) = δs[latindex](var) / lat2meters

    δs2 = []
    for i in 1:length(δs)
        if i == lonindex
            push!(δs2, Dx)
        elseif i == latindex
            push!(δs2, Dy)
        else
            push!(δs2, δs[i])
        end
    end
    return δs2
end