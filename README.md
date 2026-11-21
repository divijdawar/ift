# Interfacial Heat Transport (IFT)

IFT is a Python project for predicting phonon-mediated thermal boundary conductance at solid–solid interfaces.

## Problem and challenge

When heat crosses the boundary between two solids, some phonons transmit and others reflect. The boundary can sustain a temperature drop even when both materials conduct heat well, which makes interfaces important to chip cooling. The acoustic and diffuse mismatch models use bulk properties to estimate this resistance. Their predictions can miss effects such as inelastic phonon scattering and changes in interfacial bonding.

The Discovered Materials Grand Challenge asks for a computationally tractable model that predicts thermal boundary conductance `G(T)` and its reciprocal resistance `R_K(T) = 1/G(T)` across different interfaces using a single transferable method. The published benchmarks cover the temperature dependence of Al/Si and Al/GaN, the conductance enhancement from a TiC interlayer in Cu/diamond, and the effect of Ti or Ni adhesion layers in Au/PMMA/Si stacks. The challenge also tests whether the model improves on both mismatch models for unseen interfaces. [Challenge overview](https://discoveredmaterials.com/grand-challenges/interfacial-heat-transport/)


