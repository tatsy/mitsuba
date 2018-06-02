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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/sse.h>
#include <mitsuba/core/ssemath.h>
#include "../medium/materials.h"
#include "irrtree.h"
#include "bluenoise.h"

MTS_NAMESPACE_BEGIN

inline Float gaussian2D(Float x, Float b) {
    return (b * INV_TWOPI) * std::exp(-0.5 * b * x * x);
}

struct PlanarProjection {
    PlanarProjection(const Vector &uAxis, const Vector &vAxis, const Float &uOffset, const Float &vOffset)
        : uAxis(uAxis), vAxis(vAxis), uOffset(uOffset), vOffset(vOffset) {
    }

    Point2 map(const Point &p) {
        return Point2(uOffset + dot(Vector(p), uAxis), vOffset + dot(Vector(p) ,vAxis));
    }

    const Vector &uAxis, &vAxis;
    const Float &uOffset, &vOffset;
};

struct BSSRDFData {
    void allocate() {
        weightCoefs.resize(areaWidth * areaHeight * numGaussians);
        betas.resize(numGaussians);
    }

    const Spectrum &getWeight(const Point2 &uv, int gaussIndex) const {
        int x = (int)(uv.x * areaWidth);
        x = std::max(0, std::min(x, areaWidth - 1));
        int y = (int)(uv.y * areaHeight);
        y = std::max(0, std::min(y, areaHeight - 1));
        return weightCoefs[(y * areaWidth + x) * numGaussians + gaussIndex];
    }

    int areaWidth, areaHeight;
    int numGaussians, kernelSize;
    std::vector<Spectrum> betas;
    std::vector<Spectrum> weightCoefs;
};

/**
 * Computes the combined diffuse radiant exitance
 * caused by a number of dipole sources
 */
struct HeterogeneousBSSRDFQuery {
    inline HeterogeneousBSSRDFQuery(const BSSRDFData &bssrdf,
        const Vector &uAxis, const Vector &vAxis, const Float &uOffset,
        const Float &vOffset, const Point &p)
        : bssrdf(bssrdf), proj(uAxis, vAxis, uOffset, vOffset), result(0.0f), p(p) {
    }

#if !defined(MTS_SSE) || SPECTRUM_SAMPLES != 3
    inline void operator()(const IrradianceSample &sample) {
        Float radius = (p - sample.p).length();
        Point2 uvO = proj.map(p);
        Point2 uvI = proj.map(sample.p);
        Spectrum Rd = Spectrum(0.0);
        for (int lambda = 0; lambda < SPECTRUM_SAMPLES; lambda++) {
            Float Pi = 0.0, Po = 0.0;
            for (int h = 0; h < bssrdf.numGaussians; h++) {
                const Spectrum &wO = bssrdf.getWeight(uvO, h);
                const Spectrum &wI = bssrdf.getWeight(uvI, h);

                Float G = gaussian2D(radius, bssrdf.betas[h][lambda]);
                Po += wO[lambda] * G;
                Pi += wI[lambda] * G;
            }
            Rd[lambda] = std::sqrt(Po * Pi);
        }
        result += Rd * sample.E * sample.area;
    }
#else
    inline void operator()(const IrradianceSample &sample) {
        Float r2 = (p - sample.p).lengthSquared();
        Point2 uvO = proj.map(p);
        Point2 uvI = proj.map(sample.p);

        const __m128 _half = _mm_set1_ps(0.5f);
        const __m128 _invTwoPi = _mm_set1_ps(INV_TWOPI);
        const __m128 _rSqr = _mm_set1_ps(r2);

        __m128 _Pi = _mm_set1_ps(0.0f);
        __m128 _Po = _mm_set1_ps(0.0f);
        for (int h = 0; h < bssrdf.numGaussians; h++) {
            const Spectrum &wO = bssrdf.getWeight(uvO, h);
            const Spectrum &wI = bssrdf.getWeight(uvI, h);
            const __m128 _wO = _mm_set_ps(wO[0], wO[1], wO[2], 0.0f);
            const __m128 _wI = _mm_set_ps(wI[0], wI[1], wI[2], 0.0f);

            const Spectrum &b = bssrdf.betas[h];
            const __m128 _b = _mm_set_ps(b[0], b[1], b[2], 0.0f);

            // Gauss
            const __m128 _bHalf = _mm_mul_ps(_b, _half);
            const __m128 _expr2 = math::exp_ps(negate_ps(_mm_mul_ps(_bHalf, _rSqr)));
            const __m128 _G = _mm_mul_ps(_mm_mul_ps(_b, _invTwoPi), _expr2);

            // Profiles
            const __m128 _wOG = _mm_mul_ps(_wO, _G);
            const __m128 _wIG = _mm_mul_ps(_wI, _G);

            // Add
            _Po = _mm_add_ps(_Po, _wOG);
            _Pi = _mm_add_ps(_Pi, _wIG);
        }

        SSEVector _Rd;
        _Rd.ps = _mm_sqrt_ps(_mm_mul_ps(_Po, _Pi));

        Spectrum Rd;
        Rd[0] = _Rd.f[3];
        Rd[1] = _Rd.f[2];
        Rd[2] = _Rd.f[1];

        result += Rd * sample.E * sample.area;
    }
#endif

    inline const Spectrum &getResult() const {
        return result;
    }

    const BSSRDFData &bssrdf;
    PlanarProjection proj;
    Spectrum result;
    Point p;
};

static ref<Mutex> irrOctreeMutex = new Mutex();
static int irrOctreeIndex = 0;

/*!\plugin{dipole}{Dipole-based subsurface scattering model}
 * \parameters{
 *     \parameter{material}{\String}{
 *         Name of a material preset, see
 *         \tblref{medium-coefficients}. \default{\texttt{skin1}}
 *     }
 *     \parameter{sigmaA, sigmaS}{\Spectrum}{
 *         Absorption and scattering
 *         coefficients of the medium in inverse scene units.
 *         These parameters are mutually exclusive with \code{sigmaT} and \code{albedo}
 *         \default{configured based on \code{material}}
 *     }
 *     \parameter{sigmaT, albedo}{\Spectrum}{
 *         Extinction coefficient in inverse scene units
 *         and a (unitless) single-scattering albedo.
 *         These parameters are mutually exclusive with \code{sigmaA} and \code{sigmaS}
 *         \default{configured based on \code{material}}
 *     }
 *     \parameter{scale}{\Float}{
 *         Optional scale factor that will be applied to the \code{sigma*} parameters.
 *         It is provided for convenience when accomodating data based on different units,
 *         or to simply tweak the density of the medium. \default{1}}
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{based on \code{material}}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{based on \code{material}}}
 *     \parameter{irrSamples}{\Integer}{
 *         Number of samples to use when estimating the
 *         irradiance at a point on the surface \default{16}
 *     }
 * }
 *
 * \renderings{
 *    \rendering{The material test ball rendered with the \code{skimmilk}
 *    material preset}{subsurface_dipole.jpg}
 *    \rendering{The material test ball rendered with the \code{skin1}
 *    material preset}{subsurface_dipole_2.jpg}
 * }
 * \renderings{
 *    \rendering{\code{scale=1}}{subsurface_dipole_dragon.jpg}
 *    \rendering{\code{scale=0.2}}{subsurface_dipole_dragon2.jpg}
 *    \caption{The dragon model rendered with the \code{skin2}
 *    material preset (model courtesy of XYZ RGB). The \code{scale}
 *    parameter is useful to communicate the relative size of
 *    an object to the viewer.}
 * }

 * This plugin implements the classic dipole subsurface scattering model
 * from radiative transport and medical physics \cite{Eason1978Theory,
 * Farrell1992Diffusion} in the form proposed by Jensen et al.
 * \cite{Jensen2001Practical}. It relies on the assumption that light entering
 * a material will undergo many (i.e. hundreds) of internal scattering
 * events, such that diffusion theory becomes applicable. In this
 * case\footnote{and after making several fairly strong simplifications:
 * the geometry is assumed to be a planar half-space, and the internal
 * scattering from the material boundary is only considered approximately.}
 * a simple analytic solution of the subsurface scattering profile is available
 * that enables simulating this effect without having to account for the vast
 * numbers of internal scattering events individually.
 *
 * For each \code{dipole} instance in the scene, the plugin adds a pre-process pass
 * to the rendering that computes the irradiance on a large set of sample positions
 * spread uniformly over the surface in question. The locations of these
 * points are chosen using a technique by Bowers et al. \cite{Bowers2010Parallel}
 * that creates particularly well-distributed (blue noise) samples. Later during
 * rendering, these  illumination samples are convolved with the diffusion profile
 * using a fast hierarchical technique proposed by Jensen and Buhler \cite{Jensen2005Rapid}.
 *
 * There are two different ways of configuring the medium properties.
 * One possibility is to load a material preset
 * using the \code{material} parameter---see \tblref{medium-coefficients}
 * for details. Alternatively, when specifying parameters by hand, they
 * can either be provided using the scattering and absorption coefficients,
 * or by declaring the extinction coefficient and single scattering albedo
 * (whichever is more convenient). Mixing these parameter initialization
 * methods is not allowed.
 *
 * All scattering parameters (named \code{sigma*}) should
 * be provided in inverse scene units. For instance, when a world-space
 * distance of 1 unit corresponds to a meter, the scattering coefficents must
 * be in units of inverse meters. For convenience, the \code{scale}
 * parameter can be used to correct this. For instance, when the scene is
 * in meters and the coefficients are in inverse millimeters, set
 * \code{scale=1000}.
 *
 * Note that a subsurface integrator can be associated with an \code{id}
 * and shared by several shapes using the reference mechanism introduced in
 * \secref{format}. This can be useful when an object is made up of many
 * separate sub-shapes.
 *
 * \renderings{
 *    \medrendering{Rendered using \pluginref{dipole}}{subsurface_dipole_bad1.jpg}
 *    \medrendering{Rendered using \pluginref{homogeneous}}{subsurface_dipole_bad2.jpg}
 *    \medrendering{\code{irrSamples} set too low}{subsurface_dipole_bad3.jpg}
 *    \caption{Two problem cases that may occur when rendering with the \pluginref{dipole}:
 *     \textbf{(a)-(b)}: These two renderings show a glass ball filled with diluted milk
 *     rendered using diffusion theory and radiative transport, respectively.
 *     The former produces an incorrect result, since the assumption of
 *     many scattering events breaks down.
 *     \textbf{(c)}: When the number of irradiance samples is too low when rendering
 *     with the dipole model, the resulting noise becomes visible as ``blotchy'' artifacts
 *     in the rendering.}
 * }
 *
 * \subsubsection*{Typical material setup}
 * To create a realistic material with subsurface scattering, it is necessary
 * to associate the underlying shape with an appropriately configured surface
 * and subsurface scattering model. Both should be aware of the material's
 * index of refraction.
 *
 * Because the \pluginref{dipole} plugin is responsible for all internal
 * scattering, the surface scattering model should only account for specular
 * reflection due to the index of refraction change. There are two models
 * in Mitsuba that can do this: \pluginref{plastic} and
 * \pluginref{roughplastic} (for smooth and rough interfaces, respectively).
 * An example is given on the next page.
 * \pagebreak
 * \begin{xml}
 * <shape type="...">
 *     <subsurface type="dipole">
 *         <string name="intIOR" value="water"/>
 *         <string name="extIOR" value="air"/>
 *         <rgb name="sigmaS" value="87.2, 127.2, 143.2"/>
 *         <rgb name="sigmaA" value="1.04, 5.6, 11.6"/>
 *         <integer name="irrSamples" value="64"/>
 *     </subsurface>
 *
 *     <bsdf type="plastic">
 *         <string name="intIOR" value="water"/>
 *         <string name="extIOR" value="air"/>
 *         <!-- Note: the diffuse component must be disabled! -->
 *         <spectrum name="diffuseReflectance" value="0"/>
 *     </bsdf>
 * <shape>
 * \end{xml}
 *
 * \remarks{
 *    \item This plugin only implements the multiple scattering component of
 *    the dipole model, i.e. single scattering is omitted. Furthermore, the
 *    numerous assumptions built into the underlying theory can cause severe
 *    inaccuracies.
 *
 *    For this reason, this plugin is the right choice for making pictures
 *    that ``look nice'', but it should be avoided when the output must hold
 *    up to real-world measurements. In this case, please use participating media
 *    (\secref{media}).
 *
 *   \item It is quite important that the \code{sigma*} parameters have the right units.
 *   For instance: if the \code{sigmaT} parameter is accidentally set to a value that
 *   is too small by a factor of 1000, the plugin will attempt to create
 *   one million times as many irradiance samples, which will likely cause
 *   the rendering process to crash with an ``out of memory'' failure.
 * }
 */

class HeterogeneousBSSRDF : public Subsurface {
public:
    HeterogeneousBSSRDF(const Properties &props)
        : Subsurface(props) {
        {
            LockGuard lock(irrOctreeMutex);
            m_octreeIndex = irrOctreeIndex++;
        }

        /* How many samples should be taken when estimating
           the irradiance at a given point in the scene? */
        m_irrSamples = props.getInteger("irrSamples", 16);

        /* When estimating the irradiance at a given point,
           should indirect illumination be included in the final estimate? */
        m_irrIndirect = props.getBoolean("irrIndirect", true);

        /* Multiplicative factor, which can be used to adjust the number of
           irradiance samples */
        m_sampleMultiplier = props.getFloat("sampleMultiplier", 1.0f);

        /* Error threshold - lower means better quality */
        m_quality = props.getFloat("quality", 0.2f);

        /* Asymmetry parameter of the phase function */
        m_octreeResID = -1;

        /* Heterogeneous BSSRDF file */
        m_fileName = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));
        Log(EInfo, "Loading BSSRDF \"%s\"", m_fileName.filename().string().c_str());
        if (!fs::exists(m_fileName)) {
            Log(EError, "BSSRDF file \"%s\" could not be found!", m_fileName.string().c_str());
        }
        /* Texture coordinates transformation */
        m_uAxis = props.getVector("uAxis");
        m_vAxis = props.getVector("vAxis");
        m_uOffset = props.getFloat("uOffset");
        m_vOffset = props.getFloat("vOffset");

        /* Scaling irradiance */
        m_irrScale = props.getFloat("irrScale", 1.0);

        Float intIOR = lookupIOR(props, "intIOR", "bk7");
        Float extIOR = lookupIOR(props, "extIOR", "air");
        if (intIOR < 0 || extIOR < 0) {
            SLog(EError, "The interior and exteriro indices of refraction must be positive!");
        }

        m_scale = props.getFloat("scale", 1.0);
        m_eta = intIOR / extIOR;
    }

    HeterogeneousBSSRDF(Stream *stream, InstanceManager *manager)
     : Subsurface(stream, manager) {
        // m_sigmaS = Spectrum(stream);
        // m_sigmaA = Spectrum(stream);
        // m_g = Spectrum(stream);
        // m_eta = stream->readFloat();
        // m_sampleMultiplier = stream->readFloat();
        // m_quality = stream->readFloat();
        // m_octreeIndex = stream->readInt();
        // m_irrSamples = stream->readInt();
        // m_irrIndirect = stream->readBool();
        // m_octreeResID = -1;
        // configure();
    }

    virtual ~HeterogeneousBSSRDF() {
        if (m_octreeResID != -1)
            Scheduler::getInstance()->unregisterResource(m_octreeResID);
    }

    void bindUsedResources(ParallelProcess *proc) const {
        if (m_octreeResID != -1)
            proc->bindResource(formatString("irrOctree%i", m_octreeIndex), m_octreeResID);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        // Subsurface::serialize(stream, manager);
        // m_sigmaS.serialize(stream);
        // m_sigmaA.serialize(stream);
        // m_g.serialize(stream);
        // stream->writeFloat(m_eta);
        // stream->writeFloat(m_sampleMultiplier);
        // stream->writeFloat(m_quality);
        // stream->writeInt(m_octreeIndex);
        // stream->writeInt(m_irrSamples);
        // stream->writeBool(m_irrIndirect);
    }

    Spectrum Lo(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d, int depth) const {
        if (!m_active || dot(its.shFrame.n, d) < 0)
            return Spectrum(0.0f);

        HeterogeneousBSSRDFQuery query(bssrdf, m_uAxis, m_vAxis, m_uOffset, m_vOffset, its.p);

        m_octree->performQuery(query);
        Spectrum result(query.getResult() * (m_irrScale * m_Fdr * INV_PI));

        if (m_eta != 1.0f) {
            result *= 1.0f - fresnelDielectricExt(dot(its.shFrame.n, d), m_eta);
        }

        return result;
    }

    void configure() {
        /* Load BSSRDF data */
        std::ifstream reader(m_fileName.string().c_str(), std::ios::in | std::ios::binary);
        if (!reader.is_open()) {
            SLog(EError, "Failed to open BSSRDF File: %s", m_fileName.string().c_str());
        }

        reader.read((char*)&bssrdf.areaWidth, sizeof(int));
        reader.read((char*)&bssrdf.areaHeight, sizeof(int));
        reader.read((char*)&bssrdf.numGaussians, sizeof(int));
        reader.read((char*)&bssrdf.kernelSize, sizeof(int));
        bssrdf.allocate();

        double *buffer = new double[bssrdf.numGaussians * 3];
        for (int y = 0; y < bssrdf.areaHeight; y++) {
            for (int x = 0; x < bssrdf.areaWidth; x++) {
                reader.read((char*)buffer, sizeof(double) * bssrdf.numGaussians * 3);
                for (int h = 0; h < bssrdf.numGaussians; h++) {
                    int index = (y * bssrdf.areaWidth + x) * bssrdf.numGaussians + h;
                    const Float R = buffer[h * 3 + 0];
                    const Float G = buffer[h * 3 + 1];
                    const Float B = buffer[h * 3 + 2];
                    bssrdf.weightCoefs[index].fromLinearRGB(R, G, B);
                }
            }
        }

        reader.read((char*)buffer, sizeof(double) * bssrdf.numGaussians * 3);
        m_radius = std::numeric_limits<Float>::max();

        const Float scale2 = m_scale * m_scale;
        for (int h = 0; h < bssrdf.numGaussians; h++) {
            const Float R = buffer[h * 3 + 0] / scale2;
            const Float G = buffer[h * 3 + 1] / scale2;
            const Float B = buffer[h * 3 + 2] / scale2;
            bssrdf.betas[h].fromLinearRGB(R, G, B);

            for (int lambda = 0; lambda < SPECTRUM_SAMPLES; lambda++) {
                Float t = std::sqrt((Float)(1.0 / bssrdf.betas[h][lambda]));
                m_radius = std::min(m_radius, t);
            }
        }
        Log(EInfo, "Mean free path length: %f", m_radius);

        delete[] buffer;
        reader.close();

        /* Average diffuse reflectance due to mismatched indices of refraction */
        m_Fdr = fresnelDiffuseReflectance(1 / m_eta);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int cameraResID, int _samplerResID) {
        if (m_octree)
            return true;

        if (!scene->getIntegrator()->getClass()
                ->derivesFrom(MTS_CLASS(SamplingIntegrator)))
            Log(EError, "The dipole subsurface scattering model requires "
                "a sampling-based surface integrator!");

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Timer> timer = new Timer();

        AABB aabb;
        Float sa;

        ref<PositionSampleVector> points = new PositionSampleVector();
        /* It is necessary to increase the sampling resolution to
           prevent low-frequency noise in the output */
        Float actualRadius = m_radius / std::sqrt(m_sampleMultiplier * 20);
        blueNoisePointSet(scene, m_shapes, actualRadius, points, sa, aabb, job);

        /* 2. Gather irradiance in parallel */
        const Sensor *sensor = scene->getSensor();
        ref<IrradianceSamplingProcess> proc = new IrradianceSamplingProcess(
            points, 1024, m_irrSamples, m_irrIndirect,
            sensor->getShutterOpen() + 0.5f * sensor->getShutterOpenTime(), job);

        /* Create a sampler instance for every core */
        ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);
        int integratorResID = sched->registerResource(
            const_cast<Integrator *>(scene->getIntegrator()));

        proc->bindResource("scene", sceneResID);
        proc->bindResource("integrator", integratorResID);
        proc->bindResource("sampler", samplerResID);
        scene->bindUsedResources(proc);
        m_proc = proc;
        sched->schedule(proc);
        sched->wait(proc);
        m_proc = NULL;
        for (size_t i=0; i<samplers.size(); ++i)
            samplers[i]->decRef();

        sched->unregisterResource(samplerResID);
        sched->unregisterResource(integratorResID);
        if (proc->getReturnStatus() != ParallelProcess::ESuccess)
            return false;

        Log(EDebug, "Done gathering (took %i ms), clustering ..", timer->getMilliseconds());
        timer->reset();

        std::vector<IrradianceSample> &samples = proc->getIrradianceSampleVector()->get();
        sa /= samples.size();

        for (size_t i=0; i<samples.size(); ++i)
            samples[i].area = sa;

        m_octree = new IrradianceOctree(aabb, m_quality, samples);

        Log(EDebug, "Done clustering (took %i ms).", timer->getMilliseconds());
        m_octreeResID = Scheduler::getInstance()->registerResource(m_octree);

        return true;
    }

    void wakeup(ConfigurableObject *parent,
        std::map<std::string, SerializableObject *> &params) {
        std::string octreeName = formatString("irrOctree%i", m_octreeIndex);
        if (!m_octree.get() && params.find(octreeName) != params.end()) {
            m_octree = static_cast<IrradianceOctree *>(params[octreeName]);
            m_active = true;
        }
    }

    void cancel() {
        Scheduler::getInstance()->cancel(m_proc);
    }

    MTS_DECLARE_CLASS()
private:
    Float m_radius, m_sampleMultiplier;
    Float m_Fdr, m_quality, m_eta, m_irrScale;

    fs::path m_fileName;
    Vector m_uAxis, m_vAxis;
    Float m_uOffset, m_vOffset;
    Float m_scale;
    BSSRDFData bssrdf;

    ref<IrradianceOctree> m_octree;
    ref<ParallelProcess> m_proc;
    int m_octreeResID, m_octreeIndex;
    int m_irrSamples;
    bool m_irrIndirect;
};

MTS_IMPLEMENT_CLASS_S(HeterogeneousBSSRDF, false, Subsurface)
MTS_EXPORT_PLUGIN(HeterogeneousBSSRDF, "Heterogeneous BSSRDF");
MTS_NAMESPACE_END
