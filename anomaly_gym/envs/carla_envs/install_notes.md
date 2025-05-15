# install carla

- we use the latest version (0.9.15) since it is compatible with python 3.10
- download: wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz 
- unzip:  tar -xvzf CARLA_0.9.15.tar.gz --directory ~/CARLA_0.9.15
- install python package: pip install carla==0.9.15
- (optional) add additional modules to pythonpath: export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.9.15/PythonAPI/carla