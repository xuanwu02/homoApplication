#ifndef _SETTINGS_HPP
#define _SETTINGS_HPP

#include <string>
#include <fstream>
#include "json.hpp"

/* general */
class Settings{
public:
    int dim1;
    int dim2;
    int dim3;
    int B;
    double eb;
    inline Settings()
        : dim1(100),
          dim2(100),
          dim3(100),
          eb(1e-4)
    {}
    static Settings from_json(const std::string &fname);
};

inline void from_json(const nlohmann::json &j, Settings &s)
{
    j.at("dim1").get_to(s.dim1);
    j.at("dim2").get_to(s.dim2);
    j.at("dim3").get_to(s.dim3);
    j.at("B").get_to(s.B);
    j.at("eb").get_to(s.eb);
}

inline Settings Settings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<Settings>();
}

/* grayscott */
class gsSettings{
public:
    int L;
    int B;
    double eb;
    double F;
    double k;
    double dt;
    double Du;
    double Dv;
    int steps;
    int plotgap;
    inline gsSettings()
        : L(128),
          B(10),
          eb(1e-5),
          F(0.01),
          k(0.05),
          dt(2.0),
          Du(0.2),
          Dv(0.1),
          steps(200),
          plotgap(20)
    {}
    static gsSettings from_json(const std::string &fname);
};

inline void from_json(const nlohmann::json &j, gsSettings &s)
{
    j.at("L").get_to(s.L);
    j.at("B").get_to(s.B);
    j.at("eb").get_to(s.eb);
    j.at("F").get_to(s.F);
    j.at("k").get_to(s.k);
    j.at("dt").get_to(s.dt);
    j.at("Du").get_to(s.Du);
    j.at("Dv").get_to(s.Dv);
    j.at("steps").get_to(s.steps);
    j.at("plotgap").get_to(s.plotgap);
}

inline gsSettings gsSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<gsSettings>();
}

/* heatdis */
class htSettings{
public:
    int dim1;
    int dim2;
    int lorenzo;
    int B;
    double eb;
    float src_temp;
    float wall_temp;
    float init_temp;
    float ratio;
    int steps;
    int plotgap;
    inline htSettings()
        : dim1(100),
          dim2(100),
          lorenzo(2),
          B(8),
          eb(1e-4),
          src_temp(100.0),
          wall_temp(0.0),
          init_temp(0.0),
          ratio(0.8),
          steps(200),
          plotgap(20)
    {}
    static htSettings from_json(const std::string &fname);
};

inline void from_json(const nlohmann::json &j, htSettings &s)
{
    j.at("dim1").get_to(s.dim1);
    j.at("dim2").get_to(s.dim2);
    j.at("lorenzo").get_to(s.lorenzo);
    j.at("B").get_to(s.B);
    j.at("eb").get_to(s.eb);
    j.at("src_temp").get_to(s.src_temp);
    j.at("wall_temp").get_to(s.wall_temp);
    j.at("init_temp").get_to(s.init_temp);
    j.at("ratio").get_to(s.ratio);
    j.at("steps").get_to(s.steps);
    j.at("plotgap").get_to(s.plotgap);
}

inline htSettings htSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<htSettings>();
}

#endif