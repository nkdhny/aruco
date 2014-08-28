#ifndef HAMMINGCODE_H
#define HAMMINGCODE_H

#include <boost/noncopyable.hpp>
#include <opencv2/core/core.hpp>
namespace nkdhny {

class HammingCode: boost::noncopyable
{
private:
    static int apply_rotation(const cv::Mat & in, cv::Mat& out);
    static int distance(const cv::Mat&  bits);

    HammingCode();

public:

    /**
     * @brief rotate
     * rotate in matrix according to its normail orientation
     * @param in matrix to rotate
     * @param out rotation result
     * @return -1 if couldn't find normal orientation of matrix, 0 otherwise
     */
    static int rotate(const cv::Mat& in, cv::Mat& out);

    static int decode(const cv::Mat& in);
    /**
     * @brief encode
     * encodes given integer to its matrix repr
     * @param id integer to encode
     * @param out matrix where to store the result
     * @return -1 if unable to encode (say negative id given or something alike)
     */
    static int encode(int id, cv::Mat& out);
};

}

#endif // HAMMINGCODE_H
