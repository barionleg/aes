# flake8: noqa
import numpy as np

# Jmol colors.  See: http://jmol.sourceforge.net/jscolors/#color_U
jmol_colors = np.array([
(1.000,0.000,0.000) ,# None
(1.000,1.000,1.000), # H
(0.851,1.000,1.000), # He
(0.800,0.502,1.000), # Li
(0.761,1.000,0.000), # Be
(1.000,0.710,0.710), # B
(0.565,0.565,0.565), # C
(0.188,0.314,0.973), # N
(1.000,0.051,0.051), # O
(0.565,0.878,0.314), # F
(0.702,0.890,0.961), # Ne
(0.671,0.361,0.949), # Na
(0.541,1.000,0.000), # Mg
(0.749,0.651,0.651), # Al
(0.941,0.784,0.627), # Si
(1.000,0.502,0.000), # P
(1.000,1.000,0.188), # S
(0.122,0.941,0.122), # Cl
(0.502,0.820,0.890), # Ar
(0.561,0.251,0.831), # K
(0.239,1.000,0.000), # Ca
(0.902,0.902,0.902), # Sc
(0.749,0.761,0.780), # Ti
(0.651,0.651,0.671), # V
(0.541,0.600,0.780), # Cr
(0.612,0.478,0.780), # Mn
(0.878,0.400,0.200), # Fe
(0.941,0.565,0.627), # Co
(0.314,0.816,0.314), # Ni
(0.784,0.502,0.200), # Cu
(0.490,0.502,0.690), # Zn
(0.761,0.561,0.561), # Ga
(0.400,0.561,0.561), # Ge
(0.741,0.502,0.890), # As
(1.000,0.631,0.000), # Se
(0.651,0.161,0.161), # Br
(0.361,0.722,0.820), # Kr
(0.439,0.180,0.690), # Rb
(0.000,1.000,0.000), # Sr
(0.580,1.000,1.000), # Y
(0.580,0.878,0.878), # Zr
(0.451,0.761,0.788), # Nb
(0.329,0.710,0.710), # Mo
(0.231,0.620,0.620), # Tc
(0.141,0.561,0.561), # Ru
(0.039,0.490,0.549), # Rh
(0.000,0.412,0.522), # Pd
(0.753,0.753,0.753), # Ag
(1.000,0.851,0.561), # Cd
(0.651,0.459,0.451), # In
(0.400,0.502,0.502), # Sn
(0.620,0.388,0.710), # Sb
(0.831,0.478,0.000), # Te
(0.580,0.000,0.580), # I
(0.259,0.620,0.690), # Xe
(0.341,0.090,0.561), # Cs
(0.000,0.788,0.000), # Ba
(0.439,0.831,1.000), # La
(1.000,1.000,0.780), # Ce
(0.851,1.000,0.780), # Pr
(0.780,1.000,0.780), # Nd
(0.639,1.000,0.780), # Pm
(0.561,1.000,0.780), # Sm
(0.380,1.000,0.780), # Eu
(0.271,1.000,0.780), # Gd
(0.188,1.000,0.780), # Tb
(0.122,1.000,0.780), # Dy
(0.000,1.000,0.612), # Ho
(0.000,0.902,0.459), # Er
(0.000,0.831,0.322), # Tm
(0.000,0.749,0.220), # Yb
(0.000,0.671,0.141), # Lu
(0.302,0.761,1.000), # Hf
(0.302,0.651,1.000), # Ta
(0.129,0.580,0.839), # W
(0.149,0.490,0.671), # Re
(0.149,0.400,0.588), # Os
(0.090,0.329,0.529), # Ir
(0.816,0.816,0.878), # Pt
(1.000,0.820,0.137), # Au
(0.722,0.722,0.816), # Hg
(0.651,0.329,0.302), # Tl
(0.341,0.349,0.380), # Pb
(0.620,0.310,0.710), # Bi
(0.671,0.361,0.000), # Po
(0.459,0.310,0.271), # At
(0.259,0.510,0.588), # Rn
(0.259,0.000,0.400), # Fr
(0.000,0.490,0.000), # Ra
(0.439,0.671,0.980), # Ac
(0.000,0.729,1.000), # Th
(0.000,0.631,1.000), # Pa
(0.000,0.561,1.000), # U
(0.000,0.502,1.000), # Np
(0.000,0.420,1.000), # Pu
(0.329,0.361,0.949), # Am
(0.471,0.361,0.890), # Cm
(0.541,0.310,0.890), # Bk
(0.631,0.212,0.831), # Cf
(0.702,0.122,0.831), # Es
(0.702,0.122,0.729), # Fm
(0.702,0.051,0.651), # Md
(0.741,0.051,0.529), # No
(0.780,0.000,0.400), # Lr
(0.800,0.000,0.349), # Rf
(0.820,0.000,0.310), # Db
(0.851,0.000,0.271), # Sg
(0.878,0.000,0.220), # Bh
(0.902,0.000,0.180), # Hs
(0.922,0.000,0.149), # Mt
])

# CPK colors in units of RGB values:
cpk_colors = np.array([ 
(1.000,0.000,0.000) ,# None
(1.000,1.000,1.000) ,# H
(1.000,0.753,0.796) ,# He
(0.698,0.133,0.133) ,# Li
(1.000,0.078,0.576) ,# Be
(0.000,1.000,0.000) ,# B
(0.784,0.784,0.784) ,# C
(0.561,0.561,1.000) ,# N
(0.941,0.000,0.000) ,# O
(0.855,0.647,0.125) ,# F
(1.000,0.078,0.576) ,# Ne
(0.000,0.000,1.000) ,# Na
(0.133,0.545,0.133) ,# Mg
(0.502,0.502,0.565) ,# Al
(0.855,0.647,0.125) ,# Si
(1.000,0.647,0.000) ,# P
(1.000,0.784,0.196) ,# S
(0.000,1.000,0.000) ,# Cl
(1.000,0.078,0.576) ,# Ar
(1.000,0.078,0.576) ,# K
(0.502,0.502,0.565) ,# Ca
(1.000,0.078,0.576) ,# Sc
(0.502,0.502,0.565) ,# Ti
(1.000,0.078,0.576) ,# V
(0.502,0.502,0.565) ,# Cr
(0.502,0.502,0.565) ,# Mn
(1.000,0.647,0.000) ,# Fe
(1.000,0.078,0.576) ,# Co
(0.647,0.165,0.165) ,# Ni
(0.647,0.165,0.165) ,# Cu
(0.647,0.165,0.165) ,# Zn
(1.000,0.078,0.576) ,# Ga
(1.000,0.078,0.576) ,# Ge
(1.000,0.078,0.576) ,# As
(1.000,0.078,0.576) ,# Se
(0.647,0.165,0.165) ,# Br
(1.000,0.078,0.576) ,# Kr
(1.000,0.078,0.576) ,# Rb
(1.000,0.078,0.576) ,# Sr
(1.000,0.078,0.576) ,# Y
(1.000,0.078,0.576) ,# Zr
(1.000,0.078,0.576) ,# Nb
(1.000,0.078,0.576) ,# Mo
(1.000,0.078,0.576) ,# Tc
(1.000,0.078,0.576) ,# Ru
(1.000,0.078,0.576) ,# Rh
(1.000,0.078,0.576) ,# Pd
(0.502,0.502,0.565) ,# Ag
(1.000,0.078,0.576) ,# Cd
(1.000,0.078,0.576) ,# In
(1.000,0.078,0.576) ,# Sn
(1.000,0.078,0.576) ,# Sb
(1.000,0.078,0.576) ,# Te
(0.627,0.125,0.941) ,# I
(1.000,0.078,0.576) ,# Xe
(1.000,0.078,0.576) ,# Cs
(1.000,0.647,0.000) ,# Ba
(1.000,0.078,0.576) ,# La
(1.000,0.078,0.576) ,# Ce
(1.000,0.078,0.576) ,# Pr
(1.000,0.078,0.576) ,# Nd
(1.000,0.078,0.576) ,# Pm
(1.000,0.078,0.576) ,# Sm
(1.000,0.078,0.576) ,# Eu
(1.000,0.078,0.576) ,# Gd
(1.000,0.078,0.576) ,# Tb
(1.000,0.078,0.576) ,# Dy
(1.000,0.078,0.576) ,# Ho
(1.000,0.078,0.576) ,# Er
(1.000,0.078,0.576) ,# Tm
(1.000,0.078,0.576) ,# Yb
(1.000,0.078,0.576) ,# Lu
(1.000,0.078,0.576) ,# Hf
(1.000,0.078,0.576) ,# Ta
(1.000,0.078,0.576) ,# W
(1.000,0.078,0.576) ,# Re
(1.000,0.078,0.576) ,# Os
(1.000,0.078,0.576) ,# Ir
(1.000,0.078,0.576) ,# Pt
(0.855,0.647,0.125) ,# Au
(1.000,0.078,0.576) ,# Hg
(1.000,0.078,0.576) ,# Tl
(1.000,0.078,0.576) ,# Pb
(1.000,0.078,0.576) ,# Bi
(1.000,0.078,0.576) ,# Po
(1.000,0.078,0.576) ,# At
(1.000,1.000,1.000) ,# Rn
(1.000,1.000,1.000) ,# Fr
(1.000,1.000,1.000) ,# Ra
(1.000,1.000,1.000) ,# Ac
(1.000,0.078,0.576) ,# Th
(1.000,1.000,1.000) ,# Pa
(1.000,0.078,0.576) ,# U
(1.000,1.000,1.000) ,# Np
(1.000,1.000,1.000) ,# Pu
(1.000,1.000,1.000) ,# Am
(1.000,1.000,1.000) ,# Cm
(1.000,1.000,1.000) ,# Bk
(1.000,1.000,1.000) ,# Cf
(1.000,1.000,1.000) ,# Es
(1.000,1.000,1.000) ,# Fm
(1.000,1.000,1.000) ,# Md
(1.000,1.000,1.000) ,# No
(1.000,1.000,1.000)  # Lw
])
