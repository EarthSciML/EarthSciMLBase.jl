export partialderivatives_δxyδlonlat

# Scaling factor for converting latitude to meters
@constants lat2meters=111.32e3 * 180 / π [unit = u"m/rad"]

@constants lon2m=40075.0e3 / 2π [unit = u"m/rad"]
# Return scaling factor for converting longitude to meters at given latitude
lon2meters(lat) = lon2m * cos(lat)
# return index in vector of partial-independent variables of variable with given name
varindex(pvars::AbstractVector, varname::Symbol) = findfirst(nameof.(pvars) .== varname)

"""
$(SIGNATURES)

Return partial derivative operator transform factors corresponding
for the given partial-independent variables
after converting variables named `lon` and `lat` from degrees to x and y meters,
assuming they represent longitude and latitude on a spherical Earth.
"""
function partialderivatives_δxyδlonlat(pvars::AbstractVector; default_lat = 0.0)
    Base.depwarn(
        "EarthSciMLBase.partialderivatives_δxyδlonlat is deprecated," *
        " add coordinate transforms directly to data loaders instead",
        :partialderivatives_δxyδlonlat)
    latindex = matching_suffix_idx(pvars, :lat)
    lonindex = matching_suffix_idx(pvars, :lon)
    if length(latindex) > 1
        throw(error("Multiple variables with suffix :lat found in pvars: $(pvars[latindex])"))
    end
    if length(lonindex) > 1
        throw(error("Multiple variables with suffix :lon found in pvars: $(pvars[lonindex])"))
    end
    if length(latindex) > 0
        lat = pvars[only(latindex)]
    else
        lat = default_lat
    end

    Dict(
        only(lonindex) => 1.0 / lon2meters(lat),
        only(latindex) => 1.0 / lat2meters
    )
end
