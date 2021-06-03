import os, sys, subprocess, shutil

if __name__ == '__main__':
	# 0. Preliminary checks
	#if len(sys.argv) != 1:
	#	print("Error - run 'python build_model.py'.")
	#	sys.exit(2)
	#if not sys.argv[0] in os.listdir():
	#	print("Wrong directory - move inside '<your ecfl3 GIT repo path>/pc_simulation/' directory before running 'python build_model.py'.")
	#	sys.exit(2)
	#if not "driver.py" in os.listdir("../../cfl-core/"):
	#	print("Error: cfl-core GIT repository not found. It must in the same folder that contains ecfl3 GIT repository.")
	#	sys.exit(1)

	# 1. Remove old files
	#for entry in os.listdir():
	#	if entry.startswith("cython_model.") and (entry.endswith(".pyd") or entry.endswith(".so")):
	#		os.remove(entry)		
	#	if entry.startswith("xmc.") and (entry.endswith(".pyd") or entry.endswith(".so")):
	#		os.remove(entry)
	#	if entry in ("cython_model.pyx", "cython_model.cpp", "cython_model.html", "model_clike_nb.ipynb", "output_cfl.log", "output_cython_model.log", "output_xmc_python.log"):
	#		os.remove(entry)
	#	if entry == "build/":
	#		shutil.rmtree(entry)
	#print("Cleaning process OK.")

	# 3. Build XMC python module from model.c code using model_wrapper.c interface
	with open("output_QRDM.log", "w") as log_file:
		cp = subprocess.run("python3 setup_QRDM.py build_ext -i".split(), stdout=log_file, stderr=subprocess.STDOUT)
		if cp.returncode != 0:
			print("Error building QRDM module.")
			sys.exit(1)
		else:
			print("Build QRDM module OK.")
		
	# 5. Clean build files and move outputs in the correct folder
	#shutil.rmtree("build/")
	#os.rename("../model_specific/eCfL3_PC_simulation_draft.ipynb", "eCfL3_PC_simulation_draft.ipynb")
	
	print("\nBuild completed - OK.\n")
	sys.exit(0)
