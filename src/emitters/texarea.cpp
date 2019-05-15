/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/emitter.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/hw/gpuprogram.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{area}{Area light}
 * \icon{emitter_area}
 * \order{2}
 * \parameters{
 *     \parameter{radiance}{\Spectrum}{
 *         Specifies the emitted radiance in units of
 *         power per unit area per unit steradian.
 *     }
 *     \parameter{samplingWeight}{\Float}{
 *         Specifies the relative amount of samples
 *         allocated to this emitter. \default{1}
 *     }
 * }
 *
 * This plugin implements an area light, i.e. a light source that emits
 * diffuse illumination from the exterior of an arbitrary shape.
 * Since the emission profile of an area light is completely diffuse, it
 * has the same apparent brightness regardless of the observer's viewing
 * direction. Furthermore, since it occupies a nonzero amount of space, an
 * area light generally causes scene objects to cast soft shadows.
 *
 * When modeling scenes involving area lights, it is preferable
 * to use spheres as the emitter shapes, since they provide a
 * particularly good direct illumination sampling strategy (see
 * the \pluginref{sphere} plugin for an example).
 *
 * To create an area light source, simply instantiate the desired
 * emitter shape and specify an \code{area} instance as its child:
 *
 * \vspace{4mm}
 * \begin{xml}
 * <!-- Create a spherical light source at the origin -->
 * <shape type="sphere">
 *     <emitter type="area">
 *         <spectrum name="radiance" value="1"/>
 *     </emitter>
 * </shape>
 * \end{xml}
 */

class TexAreaLight : public Emitter {
public:
    TexAreaLight(const Properties &props) : Emitter(props) {
        m_type |= EOnSurface;

        Properties props2("scale");
        props2.setFloat("scale", Float(1.0));

        ref<Texture> texture = new ConstantSpectrumTexture(Spectrum(1.0f));
        Texture *scaledTexture = static_cast<Texture *>(PluginManager::getInstance()->createObject(MTS_CLASS(Texture), props2));
        scaledTexture->addChild(texture);
        scaledTexture->configure();
        m_radiance = static_cast<Texture2D *>(scaledTexture);
        m_power = Spectrum(0.0f); /// Don't know the power yet

        m_uscale = props.getFloat("uscale", 1.0f);
        m_vscale = props.getFloat("vscale", 1.0f);
        m_intensity = props.getSpectrum("intensity", Spectrum(1.0f));
    }

    TexAreaLight(Stream *stream, InstanceManager *manager)
        : Emitter(stream, manager) {
        m_radiance = static_cast<Texture2D *>(manager->getInstance(stream));
        m_intensity = Spectrum(stream);
        m_uscale = stream->readFloat();
        m_vscale = stream->readFloat();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Emitter::serialize(stream, manager);
        manager->serialize(stream, m_radiance.get());
        m_intensity.serialize(stream);
        stream->writeFloat(m_uscale);
        stream->writeFloat(m_vscale);
    }

    void configure() {
        Emitter::configure();
        m_invSurfaceArea = 1.0f / (m_uscale * m_vscale);
        m_power = m_radiance->getAverage() * m_intensity * m_uscale * m_vscale;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
                && (name == "radiance")) {
            m_radiance = static_cast<Texture2D *>(child);
        } else {
            Emitter::addChild(name, child);
        }
    }    

    Spectrum samplePosition(PositionSamplingRecord &pRec,
            const Point2 &sample, const Point2 *extra) const {
        const Transform &trafo = m_worldTransform->eval(pRec.time);
        
        Point2 p(2.0f * sample.x - 1.0f, 2.0f * sample.y - 1.0f);
        pRec.p = Point(trafo(Vector(p.x * m_uscale, p.y * m_vscale, 0)));
        pRec.n = trafo(Vector(0, 0, 1));
        pRec.uv = sample;
        pRec.pdf = m_invSurfaceArea;
        pRec.measure = EArea;
        return m_radiance->eval(sample) * m_intensity;
    }

    Spectrum evalPosition(const PositionSamplingRecord &pRec) const {
        return (pRec.measure == EArea) ? m_radiance->eval(pRec.uv) * m_intensity : Spectrum(0.0f);
    }

    Spectrum eval(const Intersection &its, const Vector &d) const {
        if (dot(its.shFrame.n, d) <= 1.0 - 1.0e-3) {
            return Spectrum(0.0f);
        } else {
            
            return m_radiance->eval(its) * m_intensity;
        }
    }

    Float pdfPosition(const PositionSamplingRecord &pRec) const {
        return (pRec.measure == EArea) ? m_invSurfaceArea : 0.0f;
    }

    Spectrum sampleDirection(DirectionSamplingRecord &dRec,
            PositionSamplingRecord &pRec,
            const Point2 &sample, const Point2 *extra) const {
        const Transform &trafo = m_worldTransform->eval(pRec.time);
        
        dRec.d = trafo(Vector(0, 0, 1));
        dRec.pdf = 1.0f;
        dRec.measure = EDiscrete;
        return Spectrum(1.0f);
    }

    Spectrum evalDirection(const DirectionSamplingRecord &dRec,
            const PositionSamplingRecord &pRec) const {
        Float dp = dot(dRec.d, pRec.n);

        if (dRec.measure != EDiscrete || dp < 1.0 - 1.0e-3) {
            return Spectrum(0.0f);
        }
        return Spectrum(1.0f);
    }

    Float pdfDirection(const DirectionSamplingRecord &dRec,
            const PositionSamplingRecord &pRec) const {
        Float dp = dot(dRec.d, pRec.n);

        if (dRec.measure != EDiscrete || dp < 1.0 - 1.0e-3) {
            return 0.0f;
        }
        return 1.0;
    }

    Spectrum sampleRay(Ray &ray,
            const Point2 &spatialSample,
            const Point2 &directionalSample,
            Float time) const {
        const Transform &trafo = m_worldTransform->eval(time);
        
        Point2 p(2.0f * spatialSample.x - 1.0f, 2.0f * spatialSample.y - 1.0f);
        Point origin = Point(trafo(Vector(p.x * m_uscale, p.y * m_vscale, 0)));
        Vector d = trafo(Vector(0, 0, 1));
        ray.setOrigin(origin);
        ray.setDirection(d);
        ray.setTime(time);
        return m_radiance->eval(spatialSample) * m_intensity;
    }

    Spectrum sampleDirect(DirectSamplingRecord &dRec,
            const Point2 &sample) const {
        const Transform &trafo = m_worldTransform->eval(dRec.time);

        Point p = trafo(Point(0.0f));
        Vector du = trafo(Vector(1, 0, 0));
        Vector dv = trafo(Vector(0, 1, 0));
        Vector d = trafo(Vector(0, 0, 1));
        Float distance = dot(d, dRec.ref - p);
        Float u = dot(du, dRec.ref - p) / m_uscale * 0.5f + 0.5f;
        Float v = dot(dv, dRec.ref - p) / m_vscale * 0.5f + 0.5f;
        if (distance >= 0.0f && u >= 0.0f && v >= 0.0f && u < 1.0f && v < 1.0f) {
            dRec.p = Point(p + du * (2.0f * u - 1.0f) * m_uscale + dv * (2.0f * v - 1.0f) * m_vscale);
            dRec.d = -d;
            dRec.n = Normal(d);
            dRec.dist = dot(d, dRec.ref - p);
            
            dRec.pdf = m_invSurfaceArea;
            dRec.measure = EDiscrete;
            return m_radiance->eval(Point2(u, v)) * m_intensity;
        } else {
            dRec.pdf = 0.0f;
            return Spectrum(0.0f);
        }
    }

    Float pdfDirect(const DirectSamplingRecord &dRec) const {
        const Transform &trafo = m_worldTransform->eval(dRec.time);

        Point p = trafo(Point(0.0f));
        Vector du = trafo(Vector(1, 0, 0));
        Vector dv = trafo(Vector(0, 1, 0));
        Vector d = trafo(Vector(0, 0, 1));
        Float distance = dot(d, dRec.ref - p);
        Float u = dot(du, dRec.ref - p) / m_uscale * 0.5f + 0.5f;
        Float v = dot(dv, dRec.ref - p) / m_vscale * 0.5f + 0.5f;
        if (distance >= 0.0f && u >= 0.0f && v >= 0.0f && u < 1.0f && v < 1.0f) {
            return dRec.measure == EDiscrete ? m_invSurfaceArea : 0.0f;
        } else {
            return 0.0f;
        }
    }

    AABB getAABB() const {
        return m_worldTransform->getTranslationBounds();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "TexAreaLight[" << endl
            << "  radiance = " << m_radiance.toString() << "," << endl
            << "  samplingWeight = " << m_samplingWeight << "," << endl
            << "  surfaceArea = ";
        if (m_shape)
            oss << m_shape->getSurfaceArea();
        else
            oss << "<no shape attached!>";
        oss << "," << endl
            << "  medium = " << indent(m_medium.toString()) << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()

protected:
    ref<Texture2D> m_radiance;
    Spectrum m_power, m_intensity;
    Float m_uscale, m_vscale, m_invSurfaceArea;
};

// ================ Hardware shader implementation ================

class TexAreaLightShader : public Shader {
public:
    TexAreaLightShader(Renderer *renderer, const Spectrum &radiance)
        : Shader(renderer, EEmitterShader), m_radiance(radiance) {
    }

    void resolve(const GPUProgram *program, const std::string &evalName,
            std::vector<int> &parameterIDs) const {
        parameterIDs.push_back(program->getParameterID(evalName + "_radiance", false));
    }

    void generateCode(std::ostringstream &oss, const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "uniform vec3 " << evalName << "_radiance;" << endl
            << endl
            << "vec3 " << evalName << "_area(vec2 uv) {" << endl
            << "    return " << evalName << "_radiance * pi;" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_dir(vec3 wo) {" << endl
            << "    if (cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return vec3(inv_pi);" << endl
            << "}" << endl;
    }

    void bind(GPUProgram *program, const std::vector<int> &parameterIDs,
        int &textureUnitOffset) const {
        program->setParameter(parameterIDs[0], m_radiance);
    }

    MTS_DECLARE_CLASS()
private:
    Spectrum m_radiance;
};

Shader *TexAreaLight::createShader(Renderer *renderer) const {
    return new TexAreaLightShader(renderer, m_radiance->getAverage());
}

MTS_IMPLEMENT_CLASS(TexAreaLightShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(TexAreaLight, false, Emitter)
MTS_EXPORT_PLUGIN(TexAreaLight, "Area light");
MTS_NAMESPACE_END
