# Salient object detection

A C++ implementation of paper [***Global Contrast based Salient Region detection. Ming-Ming Cheng, Niloy J. Mitra, Xiaolei Huang, Philip H. S. Torr, Shi-Min Hu. IEEE TPAMI, 2015.*** ](https://mmcheng.net/mftp/Papers/SaliencyTPAMI.pdf)

### Release
[Linux and Windows](https://github.com/Acytoo/salient-object-detection/releases)

### Data
[MSRA10K Salient Object Database](https://mmcheng.net/msra10k/)

### Building from Scratch

Have been tested on 
* Debian Testing, with g++ 8.3.0, OpenCV 3.2.0, Qt 5.11.3, OpenMP 4.5, CMake 3.13
* Windows 10, with vs2019, OpenCV 3.2.0, Qt 5, CMake 3.14

Other versions of tools might work, feel free to test it yourself.

#### Linux
You need to have a C++ compiler (supporting C++11), Cmake, Opencv abd Qt5 installed. You also need to create a separate directory for build files to prevent source tree corruption. 
Here is an example of using this project.
```
$ git clone https://github.com/Acytoo/salient-object-detection.git
$ cd salient-object-detection
$ mkdir build && cd build
$ cmake ..
$ cmake --build .
```
If nothing goes wrong, you can run the application by 
```
$ ./salient_object_detectionW.out
``` 

#### Windows
Build steps are the same as Linux, except you have to set your path of OpenCV in [CMakeLists](https://github.com/Acytoo/salient-object-detection/blob/master/CMakeLists.txt) first.
After a successful build, open generated ```.sln``` file with Visual Studio, Release + X64 mode, compile and run.


---
![demo](https://raw.githubusercontent.com/acytoo/salient-object-detection/master/resources/image/screenshot0.png)
