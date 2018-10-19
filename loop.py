from glob import glob
import os

for fn_geoint in sorted(glob('geoint_*.npz')):
	cmd = "python uhf.py " + str(fn_geoint) + " output 15 13"
	os.system(cmd)

#with open("values.txt", "a") as myfile:
#myfile.write(str(evals_alpha[nalpha-1]) + "\n")
#myfile.close()


# sumvalues = 0.
# for i in range(1,6):
# 	sumvalues += evals_alpha[nalpha-i]
# for i in range(1,4):
#   sumvalues += evals_beta[nbeta-i]

# with open("values.txt", "a") as myfile:
# myfile.write(str(sumvalues) + "\n")
# myfile.close()
