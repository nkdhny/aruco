# Aruco

This is a fork of aruco project
See http://www.uco.es/investiga/grupos/ava/node/26 and http://sourceforge.net/projects/aruco/

Basically aruco makes following:

* detects a marker on image
* reads a bit string decoded by black and white squares
* transforms bit string to a number
* calculate orientation of marker (its position and quaternion)

In this fork transformation of bit string to a number is abstract, and one can
easily change it.

## Bit string transformation

All you need is to implement class with several static methods

* `int encode(int id, cv::Mat& out)` encodes id to a bit string `out`
* `int decode(const cv::Mat& in)` decodes bit string
* `int rotate(const cv::Mat& in, cv::Mat& out)` rotates input matrix to some prefered orientation stores result of rotation in out and returns number of rotations done.
