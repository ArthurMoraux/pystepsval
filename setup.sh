#!/bin/bash -l

# Source conda initialization - adjust path based on your conda installation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization file not found. Please check your conda installation."
    exit 1
fi


# Check if the script is running inside a conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "You are not inside a conda environment. Please activate your environment first."
  exit 1
fi

# Get the current conda environment name
current_env=$(conda info --envs | grep '*' | awk '{print $1}')
echo "Current conda environment: $current_env"

# List of required packages
REQUIRED_PACKAGES=(
  xesmf
  pyproj
  pysteps
  xskillscore
  zarr
  esmpy
  setuptools
)

# Check if required packages are installed
for package in "${REQUIRED_PACKAGES[@]}"; do
  if ! conda list | grep -q "$package"; then
    echo "Error: $package is not installed in the current environment."
    exit 1
  else
    echo "$package is installed."
  fi
done

# Search for esmf.mk using find
esmf_mk_file=$(find $CONDA_PREFIX -name 'esmf.mk' 2>/dev/null)

# Check if esmf.mk was found
if [[ -z "$esmf_mk_file" ]]; then
  echo "Error: esmf.mk file not found in the conda environment."
  exit 1
else
  export ESMFMKFILE="$esmf_mk_file"
  echo "ESMFMKFILE set to $ESMFMKFILE"
fi

# Check if ESMF and ESMPy versions match using conda
esmf_version=$(conda list | grep "^esmf" | awk '{print $2}')
esmpy_version=$(conda list | grep "^esmpy" | awk '{print $2}')

if [[ "$esmf_version" != "$esmpy_version" ]]; then
  echo "Error: esmf version ($esmf_version) and ESMPy version ($esmpy_version) do not match."
  exit 1
else
  echo "esmf and ESMPy versions match: $esmpy_version"
fi

# Clone or update the FSSprob repository
# From https://zenodo.org/records/10518328
REPO_URL="https://github.com/wolfgruber/FSSprob.git"
REPO_DIR="FSSprob"
TAG="v0.2.1"

# Turn off detached head advice
git config --global advice.detachedHead false

if [ -d "$REPO_DIR" ]; then
  echo "Directory $REPO_DIR exists, pulling latest changes."
  cd "$REPO_DIR"
  git pull
  git checkout "$TAG"
  cd ..
else
  echo "Cloning $REPO_URL into $REPO_DIR."
  git clone "$REPO_URL"
  git checkout "$TAG"
fi


# Compile Fortran module with f2py
cd "$REPO_DIR"

# Check if the file fss90.cpython-311-x86_64-linux-gnu.so does not exist
if [ ! -f "fss90.cpython-311-x86_64-linux-gnu.so" ]; then
    # Compile with f2py if the file is not present
    $CONDA_PREFIX/bin/python -m numpy.f2py -c -m fss90 ./code/mod_fss.f90 --f90flags="-O3"
else
	echo "Fortran code has already been installed"
fi

cd ..
echo "Setup is complete. You can now use the package."
