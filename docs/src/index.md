```@meta
CurrentModule = EarthSciMLBase
```

# EarthSciMLBase: Utilities for Symbolic Earth Science Modeling and Machine Learning

Documentation for [EarthSciMLBase](https://github.com/EarthSciML/EarthSciMLBase.jl).

This package contains utilities for constructing Earth Science models in Julia using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl).

## Installation

```julia
using Pkg
Pkg.add("EarthSciMLBase")
```

## Feature Summary

This package contains types and functions designed to simplify the process of constructing and composing symbolically-defined Earth Science model components together.

## Feature List

* Abstract types (based on `AbstractEarthSciMLSystem`) for wrapping ModelingToolkit.jl equation systems
* Operations to compose `AbstractEarthSciMLSystem`s together using the `+` operator.
* Operations to add intitial and boundary conditions to systems and to turn ODE systems into PDE systems.
* Operations to add Advection terms to systems.

## Contributing

* Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing.

## Reproducibility
```@raw html
<details><summary>The documentation of this EarthSciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = Markdown.MD("https://github.com/EarthSciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml")
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = Markdown.MD("https://github.com/EarthSciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml")
```
```@raw html
">project</a> file.
```
