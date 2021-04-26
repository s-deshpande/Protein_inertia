from biopandas.pdb import PandasPdb
import numpy as np
from numpy import linalg as LA
import webbrowser

# Fetching the file from the rcsb site or a downloaded one. Choose either not both
pdbdf = PandasPdb().fetch_pdb('1ubq')
#pdbdf = PandasPdb().read_pdb(path = r'C:\Users\Mayth\Desktop\helix.pdb')

# Extracting the information we need into pdf (short for protein dataframe)
pdf =pdbdf.df['ATOM']

# Calculating the mass. It doesnt seem pdf files have the mass included so we have to calculate it.
# Element_dict has been made by Lukas Richter.
# Reference: https://gist.github.com/lukasrichters14/c862644d4cbcf2d67252a484b7c6049c
atom_type = pdf['element_symbol']
elements_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,
                 'OG' : 294}
molecular_mass = 0
for atom in atom_type:
    if atom in elements_dict:
        atom_mass = elements_dict.get(atom)
        molecular_mass = molecular_mass +atom_mass

# Make a new column with mass of each element
atom_mass_col = []
for atom in atom_type:
        if atom in elements_dict:
            atom_mass = elements_dict.get(atom)
            atom_mass_col.append(atom_mass)
pdf['atom_mass'] = atom_mass_col

# Finding centre of mass
# x-coordinate of centre of mass
total_numerator_x = 0
for i in pdf.index:
    numerator_x = pdf['x_coord'][i] * pdf['atom_mass'][i]
    total_numerator_x = total_numerator_x + numerator_x
x_com = total_numerator_x/molecular_mass

# y-coordinate of centre of mass
total_numerator_y = 0
for i in pdf.index:
    numerator_y = pdf['y_coord'][i] * pdf['atom_mass'][i]
    total_numerator_y = total_numerator_y + numerator_y
y_com = total_numerator_y/molecular_mass

# z-coordinate of centre of mass
total_numerator_z = 0
for i in pdf.index:
    numerator_z = pdf['z_coord'][i] * pdf['atom_mass'][i]
    total_numerator_z = total_numerator_z + numerator_z
z_com = total_numerator_z/molecular_mass


# making the moment of inertia matrix for the entire protein in the form (I11, I12, I13; I21, I22, I23; I31, I32, I33)
# Im pretty sure this aint correct but any change is pretty easy to do
# First element in row and column Ixx
I11 = 0
for i in pdf.index:
    sum1 = pdf['atom_mass'][i] * (np.square(pdf['y_coord'][i]) + np.square(pdf['z_coord'][i]))
    I11 = I11 +sum1
I11 = I11 - ((np.square(y_com) + np.square(z_com)) * molecular_mass)

# First element in column and Second in row I21
I21 = 0
for i in pdf.index:
    sum2 = pdf['atom_mass'][i] * (pdf['x_coord'][i] * pdf['y_coord'][i])
    I21 = I21 + sum2
I21 = - I21 + ((x_com * y_com) * molecular_mass)

# First element in column and Third in row I31
I31 = 0
for i in pdf.index:
    sum3 = pdf['atom_mass'][i] * (pdf['x_coord'][i] * pdf['z_coord'][i])
    I31 = I31 +sum3
I31 = - I31 + ((x_com * z_com) * molecular_mass)

# Second element in column and First in row I12
I12 = 0
for i in pdf.index:
    sum4 = pdf['atom_mass'][i] * (pdf['x_coord'][i] * pdf['y_coord'][i])
    I12 = I12 +sum4
I12 = - I12 + ((x_com * y_com) * molecular_mass)

# Second element in row and column I22
I22 = 0
for i in pdf.index:
    sum5 = pdf['atom_mass'][i] * (np.square(pdf['z_coord'][i]) + np.square(pdf['x_coord'][i]))
    I22 = I22 +sum5
I22 = I22 - ((np.square(x_com)+ np.square(z_com)) * molecular_mass)

# Second element in column and Third in row I32
I32 = 0
for i in pdf.index:
    sum6 = pdf['atom_mass'][i] * (pdf['z_coord'][i] * pdf['y_coord'][i])
    I32 = I32 +sum6
I32 = - I32 + ((z_com * y_com) * molecular_mass)

# Third element in column and First in row I13
I13 = 0
for i in pdf.index:
    sum7 = pdf['atom_mass'][i] * (pdf['z_coord'][i] * pdf['x_coord'][i])
    I13 = I13 +sum7
I13 = - I13 + ((z_com * x_com) * molecular_mass)

# Third element in column and Second in row I23
I23 = 0
for i in pdf.index:
    sum8 = pdf['atom_mass'][i] * (pdf['z_coord'][i] * pdf['y_coord'][i])
    I23 = I23 +sum8
I23 = - I23 + ((z_com * y_com) * molecular_mass)

# Third element in row and column I33
I33 = 0
for i in pdf.index:
    sum9 = pdf['atom_mass'][i] * (np.square(pdf['x_coord'][i]) + np.square(pdf['y_coord'][i]))
    I33 = I33 +sum9
I33 = I33 - ((np.square(x_com)+ np.square(y_com)) * molecular_mass)

# We use the calculated values to make the moment of inertia matrix
moi_matrix = np.array([[I11, I12, I13],
                  [I21, I22, I23],
                  [I31, I32, I33]])
moi_matrix = np.reshape(moi_matrix,(3,3))
print('moment of inertia matrix:' ,moi_matrix)

# Calculating the eigenvectors from the moment of inertia matrix
eigenval, eigenvect = LA.eig(moi_matrix)
print('eigenvalues:', eigenval)
print('original eigenvect:',eigenvect)

# Rearranging the eigenvector since python messes that up.
# This might have to be rearranged for each different pdb file since python doesn't do it correctly.
# Please visit https://www.dcode.fr/matrix-eigenvectors#comments and type/paste in the moment of inertia matrix.
# The site will now open automatically. Please check the pattern there.
# The rearrangement goes as follows. the eigenvectors of the first eigenvalue turn into first row
# Eigenvectors of second eigenvalue turn into second row and so on.
# A fix/better way to calculate eigenvectors is being worked upon. Code will be updated when that happens.
# Possible fix is to calculate eigenvectors from scratch.

a = eigenvect[0][0]
b = eigenvect[1][0]
c = eigenvect[2][0]
d = eigenvect[0][1]
e = eigenvect[1][1]
f = eigenvect[2][1]
g = eigenvect[0][2]
h = eigenvect[1][2]
i = eigenvect[2][2]

# Rearrange the matrix below. Do not mess with the thing above.

eigenvector = np.array([[d, e, f],
                  [g, h, i],
                  [-a, -b, -c]])

print('transformed eigenvector:',eigenvector)

webbrowser.open('https://www.dcode.fr/matrix-eigenvectors', new=1, autoraise=True)
# Planning to open, add and replace values directly from the website. Not implemented yet however.

# Generalising above matrix for further calculations

x00 = eigenvector[0][0]
x01 = eigenvector[0][1]
x02 = eigenvector[0][2]
y00 = eigenvector[1][0]
y01 = eigenvector[1][1]
y02 = eigenvector[1][2]
z00 = eigenvector[2][0]
z01 = eigenvector[2][1]
z02 = eigenvector[2][2]

# We need to make the centre of mass of the protein as the origin
new_x_coord = []
new_y_coord = []
new_z_coord = []
for i in pdf.index:
    x = pdf['x_coord'][i] - x_com
    y = pdf['y_coord'][i] - y_com
    z = pdf['z_coord'][i] - z_com
    new_x_coord.append(x)
    new_y_coord.append(y)
    new_z_coord.append(z)

pdf['x_coord'] = new_x_coord
pdf['y_coord'] = new_y_coord
pdf['z_coord'] = new_z_coord

# If everything up to this part is correct, we transform the coordinates by multiplying coordinate vector * eigenvector
# as such: x’= x00* x +   x01 * y +   x02 * z, y' = y00* x +   y01 * y +   y02 * z and so on
new_x_coordinate = []
new_y_coordinate = []
new_z_coordinate = []

for i in pdf.index:
    # Rotating the coordinate
    new_x = (x00 * pdf['x_coord'][i]) + (x01* pdf['y_coord'][i]) + (x02* pdf['z_coord'][i])
    new_y = y00 * pdf['x_coord'][i] + y01 * pdf['y_coord'][i] + y02 * pdf['z_coord'][i]
    new_z = z00 * pdf['x_coord'][i] + z01 * pdf['y_coord'][i] + z02 * pdf['z_coord'][i]

    # New columns can now be created in the original dataframe or the old one can be replaced with new values.
    new_x_coordinate.append(new_x)
    new_y_coordinate.append(new_y)
    new_z_coordinate.append(new_z)

# Replacing the old coordinate columns with new transformed ones
pdf['x_coord'] = new_x_coordinate
pdf['y_coord'] = new_y_coordinate
pdf['z_coord'] = new_z_coordinate

# removing the atom mass column from the dataframe
pdf = pdf.drop('atom_mass',1)



# As the protein contains a ligand the following steps will take place
ligand_df = pdbdf.df['HETATM']

# Removing water molecules/any other solvent(this list can be expanded depending on pdb file)
# This is not actually needed since every atom will be transformed including the ligand
# We can just ignore the solvents and ions. Makes the code a tad bit smaller too.
# ligand_df = ligand_df[ligand_df['residue_name'] != 'HOH']

# We need to translate the ligand into the new origin
new_x_coord = []
new_y_coord = []
new_z_coord = []
for i in ligand_df.index:
    x = ligand_df['x_coord'][i] - x_com
    y = ligand_df['y_coord'][i] - y_com
    z = ligand_df['z_coord'][i] - z_com
    new_x_coord.append(x)
    new_y_coord.append(y)
    new_z_coord.append(z)
ligand_df['x_coord'] = new_x_coord
ligand_df['y_coord'] = new_y_coord
ligand_df['z_coord'] = new_z_coord

# Transforming coordinates of ligand based on eigenvectors
new_x_coordinate_ligand = []
new_y_coordinate_ligand = []
new_z_coordinate_ligand = []

for i in ligand_df.index:
    # Rotating the coordinate
    new_ligand_x = (x00 * ligand_df['x_coord'][i]) + (x01 * ligand_df['y_coord'][i]) + (x02 * ligand_df['z_coord'][i])
    new_ligand_y = y00 * ligand_df['x_coord'][i] + y01 * ligand_df['y_coord'][i] + y02 * ligand_df['z_coord'][i]
    new_ligand_z = z00 * ligand_df['x_coord'][i] + z01 * ligand_df['y_coord'][i] + z02 * ligand_df['z_coord'][i]

    # New columns can now be created in the original dataframe or the old one can be replaced with new values.
    new_x_coordinate_ligand.append(new_ligand_x)
    new_y_coordinate_ligand.append(new_ligand_y)
    new_z_coordinate_ligand.append(new_ligand_z)

# Replacing the old coordinate columns with new transformed ones
ligand_df['x_coord'] = new_x_coordinate_ligand
ligand_df['y_coord'] = new_y_coordinate_ligand
ligand_df['z_coord'] = new_z_coordinate_ligand

# Calculating the centre of mass of the ligand. Extend this list of solvents/ions if needed. Need to uncheck any ligand
# Hashtag the HOH line if protein doesnt contain any ligand so that the code runs without an error.

ligand_df_2 = ligand_df[ligand_df['residue_name'] != 'ACT']
#ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'HOH']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'MG']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'PO4']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'AMP']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'NMN']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'SO4']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'ZN']
ligand_df_2 = ligand_df_2[ligand_df_2['residue_name'] != 'HH2']

# Calculating the molecular mass of ligand
atom_type_ligand = ligand_df_2['element_symbol']
molecular_mass_ligand = 0
for atom in atom_type_ligand:
    if atom in elements_dict:
        atom_mass_ligand = elements_dict.get(atom)
        molecular_mass_ligand = molecular_mass_ligand +atom_mass_ligand
# Make a new column with mass of each element
atom_mass_col = []
for atom in atom_type_ligand:
        if atom in elements_dict:
            atom_mass = elements_dict.get(atom)
            atom_mass_col.append(atom_mass)
ligand_df_2['atom_mass'] = atom_mass_col

# for x-coordinate centre of mass
total_numerator_ligand_x = 0
for i in ligand_df_2.index:
    numerator_x = ligand_df_2['x_coord'][i] * ligand_df_2['atom_mass'][i]
    total_numerator_ligand_x = total_numerator_ligand_x + numerator_x
x_com_ligand = total_numerator_ligand_x / molecular_mass_ligand

# for y-coordinate centre of mass
total_numerator_ligand_y = 0
for i in ligand_df_2.index:
    numerator_y = ligand_df_2['y_coord'][i] * ligand_df_2['atom_mass'][i]
    total_numerator_ligand_y = total_numerator_ligand_y + numerator_y
y_com_ligand = total_numerator_ligand_y / molecular_mass_ligand

# for z-coordinate centre of mass
total_numerator_ligand_z = 0
for i in ligand_df_2.index:
    numerator_z = ligand_df_2['z_coord'][i] * ligand_df_2['atom_mass'][i]
    total_numerator_ligand_z = total_numerator_ligand_z + numerator_z
z_com_ligand = total_numerator_ligand_z / molecular_mass_ligand

# distance of ligand COM to origin
distance_to_origin = np.sqrt(np.square(x_com_ligand)+ np.square(y_com_ligand) + np.square(z_com_ligand))

print('the centre of mass (x,y,z) of ligand is:', x_com_ligand, y_com_ligand, z_com_ligand)
print('Distance between ligand COM to origin: ', distance_to_origin, 'A')


# Exporting the transformed .pdb file
pdbdf.to_pdb(path= r'C:\Users\Mayth\Desktop\381 main report Part 1\Transformed_PDB_file.pdb',
            records= None, gz= False, append_newline= True)

# This line will run if everything has run as intended
print('Code has run successfully. A new pdb file has been created!')
