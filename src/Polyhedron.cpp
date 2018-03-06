// This file is part of eigen-cddlib.

// eigen-cddlib is free software: you can redistribute it and/or
// modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// eigen-cddlib is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with eigen-cddlib.  If not, see
// <http://www.gnu.org/licenses/>.

#include "Polyhedron.h"

struct dd_global_constants
{
    dd_global_constants()
    {
        dd_set_global_constants();
    }
    ~dd_global_constants()
    {
        dd_free_global_constants();
    }
};

const dd_global_constants singleton;

std::string to_string(dd_ErrorType err)
{
    switch(err)
    {
        case dd_DimensionTooLarge: return "dd_DimensionTooLarge";
        case dd_ImproperInputFormat: return "dd_ImproperInputFormat";
        case dd_NegativeMatrixSize: return "dd_NegativeMatrixSize";
        case dd_EmptyVrepresentation: return "dd_EmptyVrepresentation";
        case dd_EmptyHrepresentation: return "dd_EmptyHrepresentation";
        case dd_EmptyRepresentation: return "dd_EmptyRepresentation";
        case dd_IFileNotFound: return "dd_IFileNotFound";
        case dd_OFileNotOpen: return "dd_OFileNotOpen";
        case dd_NoLPObjective: return "dd_NoLPObjective";
        case dd_NoRealNumberSupport: return "dd_NoRealNumberSupport";
        case dd_NotAvailForH: return "dd_NotAvailForH";
        case dd_NotAvailForV: return "dd_NotAvailForV";
        case dd_CannotHandleLinearity: return "dd_CannotHandleLinearity";
        case dd_RowIndexOutOfRange: return "dd_RowIndexOutOfRange";
        case dd_ColIndexOutOfRange: return "dd_ColIndexOutOfRange";
        case dd_LPCycling: return "dd_LPCycling";
        case dd_NumericallyInconsistent: return "dd_NumericallyInconsistent";
        case dd_NoError: return "dd_NoError";
    }
    return "Unknown error (" + to_string(static_cast<int>(err)) + ")";
}

namespace Eigen {

std::mutex Polyhedron::mtx;

Polyhedron::~Polyhedron()
{
    if (matPtr_ != nullptr)
        dd_FreeMatrix(matPtr_);
    if (polytope_ != nullptr)
        dd_FreePolyhedra(polytope_);
}

void Polyhedron::vrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b)
{
    std::unique_lock<std::mutex> lock(mtx);
    if (!hvrep(A, b, false))
    {
        std::stringstream str;
        str << "Bad conversion from hrep to vrep.\nA:\n"
            << A << "\nb:\n"
            << b << "\nerror: "
            << to_string(err_);
        throw std::runtime_error{str.str()};
    }
}

void Polyhedron::hrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b)
{
    std::unique_lock<std::mutex> lock(mtx);
    if (!hvrep(A, b, true))
    {
        std::stringstream str;
        str << "Bad conversion from vrep to hrep.\nA:\n"
            << A << "\nb:\n"
            << b << "\nerror: "
            << to_string(err_);
        throw std::runtime_error{str.str()};
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Polyhedron::vrep() const
{
    std::unique_lock<std::mutex> lock(mtx);
    dd_MatrixPtr mat = dd_CopyGenerators(polytope_);
    return ddfMatrix2EigenMatrix(mat, true);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Polyhedron::hrep() const
{
    std::unique_lock<std::mutex> lock(mtx);
    dd_MatrixPtr mat = dd_CopyInequalities(polytope_);
    return ddfMatrix2EigenMatrix(mat, false);
}

void Polyhedron::printVrep() const
{
    std::unique_lock<std::mutex> lock(mtx);
    dd_MatrixPtr mat = dd_CopyGenerators(polytope_);
    dd_WriteMatrix(stdout, mat);
}

void Polyhedron::printHrep() const
{
    std::unique_lock<std::mutex> lock(mtx);
    dd_MatrixPtr mat = dd_CopyInequalities(polytope_);
    dd_WriteMatrix(stdout, mat);
}

/**
 * Private functions
 */

bool Polyhedron::hvrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, bool isFromGenerators)
{
    Eigen::MatrixXd cMat = concatenateMatrix(A, b, isFromGenerators);
    return doubleDescription(cMat, isFromGenerators);
}

void Polyhedron::initializeMatrixPtr(Eigen::Index rows, Eigen::Index cols, bool isFromGenerators)
{
    if (matPtr_ != nullptr)
        dd_FreeMatrix(matPtr_);

    matPtr_ = dd_CreateMatrix(rows, cols);
    matPtr_->representation = (isFromGenerators ? dd_Generator : dd_Inequality);
}

bool Polyhedron::doubleDescription(const Eigen::MatrixXd& matrix, bool isFromGenerators)
{
    initializeMatrixPtr(matrix.rows(), matrix.cols(), isFromGenerators);

    for (auto row = 0; row < matrix.rows(); ++row)
        for (auto col = 0; col < matrix.cols(); ++col)
            matPtr_->matrix[row][col][0] = matrix(row, col);

    if (polytope_ != nullptr)
        dd_FreePolyhedra(polytope_);

    polytope_ = dd_DDMatrix2Poly(matPtr_, &err_);
    return (err_ == dd_NoError) ? true : false;
}

Eigen::MatrixXd Polyhedron::concatenateMatrix(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, bool isFromGenerators)
{
    double sign = (isFromGenerators ? 1 : -1);
    Eigen::MatrixXd mat(A.rows(), A.cols() + 1);
    mat.col(0) = b;
    mat.block(0, 1, A.rows(), A.cols()).noalias() = sign * A;
    return mat;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Polyhedron::ddfMatrix2EigenMatrix(const dd_MatrixPtr mat, bool isOuputVRep) const
{
    double sign = (isOuputVRep ? 1 : -1);
    auto rows = mat->rowsize;
    auto cols = mat->colsize;
    Eigen::MatrixXd mOut(rows, cols - 1);
    Eigen::VectorXd vOut(rows);
    for (auto row = 0; row < rows; ++row) {
        vOut(row) = mat->matrix[row][0][0];
        for (auto col = 1; col < cols; ++col)
            mOut(row, col - 1) = sign * mat->matrix[row][col][0];
    }

    return std::make_pair(mOut, vOut);
}

PolyhedronHRep::PolyhedronHRep(std::pair<MatrixXd, VectorXd> hrep) :
    hrepA_{std::move(hrep.first)}, hrepB_{std::move(hrep.second)}
{
    checkAndNormalize();
}

PolyhedronHRep& PolyhedronHRep::operator=(std::pair<MatrixXd, VectorXd> hrep)
{
    hrepA_ = std::move(hrep.first);
    hrepB_ = std::move(hrep.second);
    checkAndNormalize();
    return *this;
}

PolyhedronHRep::PolyhedronHRep(MatrixXd A, VectorXd B) :
    hrepA_{std::move(A)}, hrepB_{std::move(B)}
{
    checkAndNormalize();
}

PolyhedronHRep::PolyhedronHRep(const MatrixXd& A, const VectorXd& B) :
    hrepA_{A}, hrepB_{B}
{
    checkAndNormalize();
}

bool PolyhedronHRep::contains(const Eigen::VectorXd& x) const
{
    checkX(x);
    return ((hrepA_ * x).array() <= hrepB_.array()).all();
}

double PolyhedronHRep::evaluate(const VectorXd &x) const
{
    checkX(x);
    return ((hrepA_ * x) - hrepB_).maxCoeff();
}

VectorXd PolyhedronHRep::rayIntersection(const VectorXd& origin, const VectorXd& direction, double shrinkBorderBy) const
{
    const auto eval = evaluate(origin);
    if(std::abs(eval) < 1e-10)
    {
        // origin is on a border surface
        return origin;
    }
    checkX(direction, "direction");
    shrinkBorderBy = std::abs(shrinkBorderBy);
    if(eval < 0)
    {
        //origin is in the polyhedron

        // a^T * x - b = - shrinkBorderBy
        // x = origin + j * direction
        // a^T * (origin + j * direction) - b = - shrinkBorderBy
        // a^T * origin + a^T * j * direction - b = - shrinkBorderBy
        // a^T * j * direction - b = b - shrinkBorderBy - a^T * origin
        // j = (b - shrinkBorderBy - a^T * origin) / (a^T * direction)
        // -> intersection = min pos j
        // if all j < 0 -> no intersection -> return origin
        double j = std::numeric_limits<double>::infinity();
        for(int  i = 0; i < hrepA_.rows(); ++i)
        {
            const double j_ = (hrepB_(i) - shrinkBorderBy - hrepA_.row(i) * origin) / (hrepA_.row(i) * direction);
            if(j_ > 0 && std::isfinite(j_) && j_ < j)
            {
                j = j_;
            }
        }
        if(j == std::numeric_limits<double>::infinity())
        {
            //the polyhedron is open and the ray heads in this direction
            return origin;
        }
        return origin + j * direction;
    }

    //origin is outside of the polyhedron
    double j = std::numeric_limits<double>::infinity();
    Eigen::VectorXd result = origin;
    for(int  i = 0; i < hrepA_.rows(); ++i)
    {
        const double j_ = (hrepB_(i) - shrinkBorderBy - hrepA_.row(i) * origin) / (hrepA_.row(i) * direction);
        if(j_ > 0 && std::isfinite(j_) && j_ < j)
        {
            //this intersection point has less distance than the current intersection
            const Eigen::VectorXd intersection = origin + j_ * direction;
            if(std::abs(evaluate(origin) + shrinkBorderBy) < 1e-10)
            {
                //this intersection has the specified distance
                //use it
                result = intersection;
                j = j_;
            }

        }
    }
    return result;
}

const MatrixXd &PolyhedronHRep::A() const
{
    return hrepA_;
}

const VectorXd &PolyhedronHRep::B() const
{
    return hrepB_;
}

std::pair<MatrixXd, VectorXd> PolyhedronHRep::hrep() const
{
    return {hrepA_, hrepB_};
}

std::pair<MatrixXd, VectorXd> PolyhedronHRep::vrep() const
{
    Polyhedron p;
    p.vrep(hrepA_, hrepB_);
    return p.vrep();
}

void PolyhedronHRep::checkAndNormalize()
{
    if (hrepA_.rows() != hrepB_.rows())
        throw std::invalid_argument(
                "A and b must have the same number of rows! They have " + to_string(hrepA_.rows()) +
                " and " + to_string(hrepB_.rows()) + " rows."
                );
    for(int i = 0; i < hrepA_.rows(); ++i)
    {
        const auto len = hrepA_.row(i).lpNorm<2>();
        hrepA_.row(i) /= len;
        hrepB_.row(i) /= len;
    }
}

void PolyhedronHRep::checkX(const VectorXd &x, const char* name) const
{
    if (x.rows() != hrepA_.cols())
    {
        throw std::invalid_argument{
            std::string{name} + " has to be a col vector of size " + to_string(hrepA_.cols()) +
            ". " + std::string{name} + " has size " + to_string(x.rows()) + "X" + to_string(x.cols())
        };
    }
}

} // namespace Eigen
