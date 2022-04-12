# SimBA-PCA

This is a rewrite and extension of the [simple_blackbox_attack](https://github.com/cg563/simple-blackbox-attack) 
(SimBA) repo which implements the paper "[Simple Black-box Adversarial Attacks](https://arxiv.org/abs/1905.07121)".

It supports the section "On the importance of input-specific priors" in the ICLR 2022 paper "[Attacking deep
networks with surrogate-based adversarial black-box methods is easy](https://arxiv.org/abs/2203.08725)", by Nicholas A.
Lord, Romain Mueller, and Luca Bertinetto. That paper's main attack method is implemented in the [GFCS repo](
https://github.com/fiveai/GFCS). SimBA-PCA is required to perform the [reproduction](
https://github.com/fiveai/GFCS/blob/main/fig4_input-specific_priors.md) of the results in the aforementioned section of
the paper, using scripts provided in the GFCS repo.

The extended main attack, "Mufasa", allows for the straightforward definition and incorporation of alternative 
SimBA attack-basis choices (as well as supporting various other options). We also provide supporting methods to compute 
those attack bases through singular value decomposition (equivalent to PCA) of collections of samples. These samples 
will typically represent adversarial perturbations (i.e. network features/sensitivity directions), but can represent 
whatever the user wants them to.

## Usage

Main scripts intended to be run directly live in scripts_core. Each one supports a help (-h, --help) option explaining 
its parameters.

mufasa.py performs the main SimBA-style attack. When this is run using the "pixel" or "dct" attack basis options, it is 
effectively SimBA (with optional enhancements). These are formulaic basis choices that don't require any offline 
analysis, and Mufasa can be run directly on the attacked set under these settings.

When the "enlo" basis choice is activated, Mufasa instead looks to read its basis vectors from disc. The typical flow 
for generating and using basis vectors in that manner is 

generate_adversaries.py &rarr; dominant_directions.py &rarr; mufasa.py,

where:
- generate_adversaries produces a collection of samples;
- dominant_directions assembles them into a data matrix and decomposes it, producing the basis; and
- mufasa reads that basis and uses it in a SimBA-style attack.

For sample command-line calls which illustrate the entire sequence, please see the "On the importance of input-specific 
priors" reproduction section in the GFCS codebase, as detailed and linked above. That repo also contains utilities
for plotting the results.

## Setup

To create the conda environment for the first time, run the following from the top-level project directory:
```
conda env create -f environment.yml
```

(Note that your system must support CUDA 10.2.)

To activate the environment and set paths up for the interpreter session, run these:
```
conda activate mufasa
pip install -e .
```


