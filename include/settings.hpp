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
          B(8),
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
    int offset;
    double criteria;
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
          plotgap(20),
          offset(1),
          criteria(1e-6)
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
    j.at("offset").get_to(s.offset);
    j.at("criteria").get_to(s.criteria);
}

inline gsSettings gsSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<gsSettings>();
}

/* heatdis2D */
class ht2DSettings{
public:
    int dim1, dim2;
    std::string type;
    int dim;
    int B;
    double eb;
    float src_temp;
    float wall_temp;
    float init_temp;
    float ratio;
    int steps;
    int plotgap;
    int offset;
    double criteria;
    inline ht2DSettings()
        : dim1(100),
          dim2(100),
          type("p"),
          dim(2),
          B(8),
          eb(1e-4),
          src_temp(100.0),
          wall_temp(0.0),
          init_temp(0.0),
          ratio(0.8),
          steps(200),
          plotgap(20),
          offset(1),
          criteria(1e-4)
    {}
    static ht2DSettings from_json(const std::string &fname);
};

inline void from_json(const nlohmann::json &j, ht2DSettings &s)
{
    j.at("dim1").get_to(s.dim1);
    j.at("dim2").get_to(s.dim2);
    j.at("type").get_to(s.type);
    j.at("dim").get_to(s.dim);
    j.at("B").get_to(s.B);
    j.at("eb").get_to(s.eb);
    j.at("src_temp").get_to(s.src_temp);
    j.at("wall_temp").get_to(s.wall_temp);
    j.at("init_temp").get_to(s.init_temp);
    j.at("ratio").get_to(s.ratio);
    j.at("steps").get_to(s.steps);
    j.at("plotgap").get_to(s.plotgap);
    j.at("offset").get_to(s.offset);
    j.at("criteria").get_to(s.criteria);
}

inline ht2DSettings ht2DSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<ht2DSettings>();
}

/* heatdis3D */
class ht3DSettings{
public:
    int dim1, dim2, dim3;
    std::string type;
    int dim;
    int B;
    double eb;
    float alpha;
    float T_top;
    float T_bott;
    float T_wall;
    float T_init;
    int steps;
    int plotgap;
    int offset;
    double criteria;
    inline ht3DSettings()
        : dim1(64),
          dim2(64),
          dim3(64),
          type("p"),
          dim(3),
          B(8),
          eb(1e-4),
          alpha(0.01),
          T_top(-20.0),
          T_bott(20.0),
          T_wall(0.0),
          T_init(0.0),
          steps(200),
          plotgap(20),
          offset(1),
          criteria(1e-4)
    {}
    static ht3DSettings from_json(const std::string &fname);
};

inline void from_json(const nlohmann::json &j, ht3DSettings &s)
{
    j.at("dim1").get_to(s.dim1);
    j.at("dim2").get_to(s.dim2);
    j.at("dim3").get_to(s.dim3);
    j.at("type").get_to(s.type);
    j.at("dim").get_to(s.dim);
    j.at("B").get_to(s.B);
    j.at("eb").get_to(s.eb);
    j.at("alpha").get_to(s.alpha);
    j.at("T_top").get_to(s.T_top);
    j.at("T_bott").get_to(s.T_bott);
    j.at("T_wall").get_to(s.T_wall);
    j.at("T_init").get_to(s.T_init);
    j.at("steps").get_to(s.steps);
    j.at("plotgap").get_to(s.plotgap);
    j.at("offset").get_to(s.offset);
    j.at("criteria").get_to(s.criteria);
}

inline ht3DSettings ht3DSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;
    ifs >> j;
    return j.get<ht3DSettings>();
}

#endif