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

#define BOOST_TEST_MODULE TestPolyhedron
#include <Eigen/Core>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include <utility>

#include "Polyhedron.h"

struct Rep {
    Rep()
        : AVrep(4, 3)
        , AHrep(4, 3)
        , bVrep(4)
        , bHrep(4)
    {
        AVrep << 1, 1, 2,
            1, -1, 2,
            -1, -1, 2,
            -1, 1, 2; // Friction cone inequality * 2
        AHrep << -2, 0, -1,
            0, -2, -1,
            2, 0, -1,
            0, 2, -1;
        bVrep << 0, 0, 0, 0;
        bHrep << 0, 0, 0, 0;
    }

    Eigen::MatrixXd AVrep, AHrep;
    Eigen::VectorXd bVrep, bHrep;
};

BOOST_FIXTURE_TEST_CASE(Vrep2Hrep, Rep)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::Polyhedron poly;
    poly.hrep(AVrep, bVrep);
    auto hrep = poly.hrep();
    BOOST_CHECK(AHrep.isApprox(hrep.first));
    BOOST_CHECK(bHrep.isApprox(hrep.second));
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Wall time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "ms" << std::endl;

    poly.printHrep();
}

BOOST_FIXTURE_TEST_CASE(Hrep2Vrep, Rep)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::Polyhedron poly;
    poly.vrep(AHrep, bHrep);
    auto vrep = poly.vrep();
    BOOST_CHECK(AVrep.isApprox(vrep.first));
    BOOST_CHECK(bVrep.isApprox(vrep.second));
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Wall time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "ms" << std::endl;

    poly.printVrep();
}

BOOST_FIXTURE_TEST_CASE(Contains, Rep)
{
    Eigen::Polyhedron poly;
    poly.vrep(AHrep, bHrep);
    Eigen::PolyhedronHRep h{poly.hrep()};
    for(int i = 0; i < AVrep.rows(); ++i)
        BOOST_CHECK(h.contains(AVrep.row(i).transpose()));

    //check plane at z=2 contained
    for(int x = -100; x <= 100; ++x)
    {
        for(int y = -100; y <= 100; ++y)
        {
            BOOST_CHECK(h.contains(Eigen::Vector3d{0.01*x, 0.01*y, 2}));
        }
    }
    //check plane at z=2 excluded
    for(int x = 1; x <= 100; ++x)
    {
        for(int y = 1; y <= 100; ++y)
        {
            BOOST_CHECK(!h.contains(Eigen::Vector3d{1+0.01*x, 0.01*y, 2}));
            BOOST_CHECK(!h.contains(Eigen::Vector3d{0.01*x, 1+0.01*y, 2}));
            BOOST_CHECK(!h.contains(Eigen::Vector3d{-1-0.01*x, 0.01*y, 2}));
            BOOST_CHECK(!h.contains(Eigen::Vector3d{0.01*x, -1-0.01*y, 2}));
        }
    }
    BOOST_CHECK(h.contains(Eigen::Vector3d{0,0,0}));
    BOOST_CHECK(h.contains(Eigen::Vector3d{1, 0, 2}));
    BOOST_CHECK(h.contains(Eigen::Vector3d{0,0,1}));
    BOOST_CHECK(!h.contains(Eigen::Vector3d{0,0,-1}));
}

BOOST_FIXTURE_TEST_CASE(Evaluate, Rep)
{
    Eigen::Polyhedron poly;
    poly.vrep(AHrep, bHrep);
    Eigen::PolyhedronHRep h{poly.hrep()};
    for(int i = 0; i < AVrep.rows(); ++i)
        BOOST_CHECK_EQUAL(h.evaluate(AVrep.row(i).transpose()), 0);
    //check edges at z = 2
    for(int i = -100; i <= 100; ++i)
    {
        BOOST_CHECK_EQUAL(h.evaluate(Eigen::Vector3d{1,0.01*i,2}), 0);
        BOOST_CHECK_EQUAL(h.evaluate(Eigen::Vector3d{-1,0.01*i,2}), 0);
        BOOST_CHECK_EQUAL(h.evaluate(Eigen::Vector3d{0.01*i,1,2}), 0);
        BOOST_CHECK_EQUAL(h.evaluate(Eigen::Vector3d{0.01*i,-1,2}), 0);
    }
    //check z-axis (in poly)
    BOOST_CHECK_EQUAL(h.evaluate(Eigen::Vector3d{0,0,0}), 0);
    for(int i = 1; i <= 100; ++i)
    {
        const auto z=0.1*i;
        const auto x = 0.5 * z;
        BOOST_CHECK_GT(h.evaluate(Eigen::Vector3d{0,0,-z}), 0);
        BOOST_CHECK_CLOSE(h.evaluate(Eigen::Vector3d{0,0,z}), -z*x/std::hypot(z,x), 1e-10);
    }
}
