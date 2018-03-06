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

#pragma once

#include "typedefs.h"
#include <Eigen/Core>
#include <atomic>
#include <cdd/setoper.h> // Must be included before cdd.h (wtf)
#include <cdd/cdd.h>
#include <mutex>
#include <utility>

namespace Eigen {
/** Wrapper of Convex Polyhedron
 * This class aims to translate eigen matrix into cddlib matrix.
 * It automatically transforms a v-polyhedron into an h-polyhedron and vice-versa.
 */
class Polyhedron {
public:
    /** Free the pointers.*/
    ~Polyhedron();

    /** Treat the inputs as a H-representation and compute its V-representation.
     * H-polyhedron is such that \f$ Ax \leq b \f$.
     * \param A The matrix part of the representation of the polyhedron.
     * \param b The vector part of the representation of the polyhedron.
     */
    void vrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    /** Treat the inputs as a V-representation and compute its H-representation
     * V-polyhedron is such that \f$ A = [v r]^T, b=[1^T 0^T]^T \f$
     * with A composed of \f$ v \f$, the vertices, \f$ r \f$, the rays
     * and b is a vector which is 1 for vertices and 0 for rays.
     * \param A The matrix part of the representation of the polyhedron.
     * \param b The vector part of the representation of the polyhedron.
     */
    void hrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    /** Get the V-representation of the polyhedron
     * V-polyhedron is such that \f$ A = [v^T r^T]^T, b=[1^T 0^T]^T \f$
     * with A composed of \f$ v \f$, the vertices, \f$ r \f$, the rays
     * and b is a vector which is 1 for vertices and 0 for rays.
     * \return Pair of vertices and rays matrix and identification vector of vertices and rays for the V-representation
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> vrep() const;
    /** Get the H-representation of the polyhedron
     * H-polyhedron is such that \f$ Ax \leq b \f$.
     * \return Pair of inequality matrix and inequality vector for the H-representation
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> hrep() const;

    /** Print the H-representation of the polyhedron */
    void printHrep() const;
    /** Print the V-representation of the polyhedron */
    void printVrep() const;

private:
    bool hvrep(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, bool isFromGenerators);
    void initializeMatrixPtr(Eigen::Index rows, Eigen::Index cols, bool isFromGenerators);
    bool doubleDescription(const Eigen::MatrixXd& matrix, bool isFromGenerators);
    Eigen::MatrixXd concatenateMatrix(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, bool isFromGenerators);
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> ddfMatrix2EigenMatrix(const dd_MatrixPtr mat, bool isOuputVRep) const;

private:
    dd_MatrixPtr matPtr_{nullptr};
    dd_PolyhedraPtr polytope_{nullptr};
    dd_ErrorType err_;
private:
    static std::mutex mtx;
};

/**
 * @brief Wraps a Polyhedron hrep and offers functions to check whether a point is contained or
 * evaluate the hrep for a point and get distance information.
 */
class PolyhedronHRep
{
public:
    /**
     * @brief Construct a PolyhedronHRep from the given hrep.
     * @param hrep The hrep as pair of (A, b)
     */
    PolyhedronHRep(std::pair<Eigen::MatrixXd, Eigen::VectorXd> hrep);
    /**
     * @brief Construct a PolyhedronHRep from the given hrep.
     * @param A The hreps matrix
     * @param B The hreps vector
     */
    PolyhedronHRep(Eigen::MatrixXd A = {}, Eigen::VectorXd B = {});
    /**
     * @brief Construct a PolyhedronHRep from the given hrep.
     * @param A The hreps matrix
     * @param B The hreps vector
     */
    PolyhedronHRep(const Eigen::MatrixXd& A, const Eigen::VectorXd& B);

    /** Default copy ctor */
    PolyhedronHRep(const PolyhedronHRep&) = default;
    /** Default move ctor */
    PolyhedronHRep(PolyhedronHRep&&) = default;

    /**
     * @brief Assign the given hrep.
     * @param hrep The hrep as pair of (A, b)
     */
    PolyhedronHRep& operator=(std::pair<Eigen::MatrixXd, Eigen::VectorXd> hrep);
    /** Default copy assign */
    PolyhedronHRep& operator=(const PolyhedronHRep&) = default;
    /** Default move assign */
    PolyhedronHRep& operator=(PolyhedronHRep&&) = default;

    /**
     * @param x The point to check.
     * @return True if the point is contained in the hrep.
     */
    bool contains(const Eigen::VectorXd& x) const;

    /**
     * @param x The point to evaluate.
     * @return If the point is contained: minus the distance to the Polyhedrons surface.
     * Otherwise a lower bound for the distance to the Polyhedrons surface.
     */
    double evaluate(const Eigen::VectorXd& x) const;

    /**
     * @param origin The Ray origin
     * @param direction The Ray direction
     * @return The intersection between the given ray and polyhedron or origin if there is no intersection,
     */
    Eigen::VectorXd rayIntersection(const Eigen::VectorXd& origin, const Eigen::VectorXd& direction, double shrinkBorderBy = 0) const;

    const Eigen::MatrixXd& A() const;
    const Eigen::VectorXd& B() const;
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> hrep() const;
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> vrep() const;
private:
    void checkAndNormalize();
    void checkX(const Eigen::VectorXd& x, const char *name = "X") const;

    Eigen::MatrixXd hrepA_;
    Eigen::VectorXd hrepB_;
};


} // namespace Eigen
