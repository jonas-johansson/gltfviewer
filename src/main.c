// gltfviewer — a small native glTF 2.0 / GLB viewer for Linux (SDL3 + OpenGL 3.3)
//
// Features: metallic-roughness PBR (plus spec-gloss fallback), normal/occlusion/emissive
// maps, alpha mask/blend, double-sided, KHR_materials_unlit, emissive_strength,
// node animation (step/linear/cubic), skinning, orbit camera, ground grid, drag & drop.

#define _GNU_SOURCE
#include <SDL3/SDL.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_WARNINGS
#include "stb_image.h"
#include "stb_easy_font.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI_F 3.14159265358979f
#define FOV_Y (45.0f * PI_F / 180.0f)
#define USER_EVENT_OPEN_FILE 1

// ---------------------------------------------------------------------------------------------
// Math

typedef struct { float x, y, z; } vec3;
typedef struct { float m[16]; } mat4; // column-major

static vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static vec3 v3_scale(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float v3_len(vec3 a) { return sqrtf(v3_dot(a, a)); }
static vec3 v3_norm(vec3 a) {
    float l = v3_len(a);
    return l > 1e-20f ? v3_scale(a, 1.0f / l) : v3(0, 1, 0);
}

static mat4 m4_identity(void) {
    mat4 r = {{0}};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}

static mat4 m4_trs(const float t[3], const float q[4], const float s[3]) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    mat4 r;
    r.m[0] = (1 - 2 * (yy + zz)) * s[0];
    r.m[1] = 2 * (xy + wz) * s[0];
    r.m[2] = 2 * (xz - wy) * s[0];
    r.m[3] = 0;
    r.m[4] = 2 * (xy - wz) * s[1];
    r.m[5] = (1 - 2 * (xx + zz)) * s[1];
    r.m[6] = 2 * (yz + wx) * s[1];
    r.m[7] = 0;
    r.m[8] = 2 * (xz + wy) * s[2];
    r.m[9] = 2 * (yz - wx) * s[2];
    r.m[10] = (1 - 2 * (xx + yy)) * s[2];
    r.m[11] = 0;
    r.m[12] = t[0];
    r.m[13] = t[1];
    r.m[14] = t[2];
    r.m[15] = 1;
    return r;
}

static mat4 m4_perspective(float fovy, float aspect, float n, float f) {
    mat4 r = {{0}};
    float t = 1.0f / tanf(fovy * 0.5f);
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = (f + n) / (n - f);
    r.m[11] = -1;
    r.m[14] = 2 * f * n / (n - f);
    return r;
}

static mat4 m4_lookat(vec3 eye, vec3 center, vec3 up) {
    vec3 f = v3_norm(v3_sub(center, eye));
    vec3 s = v3_norm(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    mat4 r = m4_identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -v3_dot(s, eye);
    r.m[13] = -v3_dot(u, eye);
    r.m[14] = v3_dot(f, eye);
    return r;
}

static vec3 m4_point(const mat4 *m, vec3 p) {
    return v3(m->m[0] * p.x + m->m[4] * p.y + m->m[8] * p.z + m->m[12],
              m->m[1] * p.x + m->m[5] * p.y + m->m[9] * p.z + m->m[13],
              m->m[2] * p.x + m->m[6] * p.y + m->m[10] * p.z + m->m[14]);
}

// Inverse-transpose of the upper 3x3 (normal matrix). Returns the determinant.
static float m4_normal_matrix(const mat4 *m, float out[9]) {
    vec3 c0 = v3(m->m[0], m->m[1], m->m[2]);
    vec3 c1 = v3(m->m[4], m->m[5], m->m[6]);
    vec3 c2 = v3(m->m[8], m->m[9], m->m[10]);
    vec3 a = v3_cross(c1, c2), b = v3_cross(c2, c0), c = v3_cross(c0, c1);
    float det = v3_dot(c0, a);
    float inv = fabsf(det) > 1e-30f ? 1.0f / det : 0.0f;
    out[0] = a.x * inv; out[1] = a.y * inv; out[2] = a.z * inv;
    out[3] = b.x * inv; out[4] = b.y * inv; out[5] = b.z * inv;
    out[6] = c.x * inv; out[7] = c.y * inv; out[8] = c.z * inv;
    return det;
}

static void quat_slerp(const float *a, const float *b, float t, float *out) {
    float bb[4] = {b[0], b[1], b[2], b[3]};
    float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (d < 0) {
        d = -d;
        for (int i = 0; i < 4; i++) bb[i] = -bb[i];
    }
    float wa, wb;
    if (d > 0.9995f) {
        wa = 1 - t;
        wb = t;
    } else {
        float th = acosf(d), s = sinf(th);
        wa = sinf((1 - t) * th) / s;
        wb = sinf(t * th) / s;
    }
    float l = 0;
    for (int i = 0; i < 4; i++) {
        out[i] = wa * a[i] + wb * bb[i];
        l += out[i] * out[i];
    }
    l = l > 0 ? 1.0f / sqrtf(l) : 1.0f;
    for (int i = 0; i < 4; i++) out[i] *= l;
}

// ---------------------------------------------------------------------------------------------
// Shaders

static const char *MESH_VS =
    "#version 330 core\n"
    "layout(location=0) in vec3 a_pos;\n"
    "layout(location=1) in vec3 a_nrm;\n"
    "layout(location=2) in vec2 a_uv0;\n"
    "layout(location=3) in vec2 a_uv1;\n"
    "layout(location=4) in vec4 a_col;\n"
    "layout(location=5) in vec4 a_joints;\n"
    "layout(location=6) in vec4 a_weights;\n"
    "layout(location=7) in vec4 a_tan;\n"
    "uniform mat4 u_model;\n"
    "uniform mat3 u_nmat;\n"
    "uniform mat4 u_viewproj;\n"
    "uniform int u_skinned;\n"
    "uniform samplerBuffer u_joint_tex;\n"
    "out vec3 v_pos; out vec3 v_nrm; out vec4 v_tan; out vec2 v_uv0; out vec2 v_uv1; out vec4 v_col;\n"
    "mat4 joint_mat(float fi) {\n"
    "  int b = int(fi) * 4;\n"
    "  return mat4(texelFetch(u_joint_tex, b), texelFetch(u_joint_tex, b + 1),\n"
    "              texelFetch(u_joint_tex, b + 2), texelFetch(u_joint_tex, b + 3));\n"
    "}\n"
    "void main() {\n"
    "  vec4 wp; vec3 n; vec3 t;\n"
    "  if (u_skinned == 1) {\n"
    "    mat4 s = a_weights.x * joint_mat(a_joints.x) + a_weights.y * joint_mat(a_joints.y)\n"
    "           + a_weights.z * joint_mat(a_joints.z) + a_weights.w * joint_mat(a_joints.w);\n"
    "    wp = s * vec4(a_pos, 1.0);\n"
    "    mat3 s3 = mat3(s);\n"
    "    n = s3 * a_nrm; t = s3 * a_tan.xyz;\n"
    "  } else {\n"
    "    wp = u_model * vec4(a_pos, 1.0);\n"
    "    n = u_nmat * a_nrm; t = mat3(u_model) * a_tan.xyz;\n"
    "  }\n"
    "  v_pos = wp.xyz; v_nrm = n; v_tan = vec4(t, a_tan.w);\n"
    "  v_uv0 = a_uv0; v_uv1 = a_uv1; v_col = a_col;\n"
    "  gl_Position = u_viewproj * wp;\n"
    "}\n";

static const char *MESH_FS =
    "#version 330 core\n"
    "in vec3 v_pos; in vec3 v_nrm; in vec4 v_tan; in vec2 v_uv0; in vec2 v_uv1; in vec4 v_col;\n"
    "out vec4 o_color;\n"
    "uniform vec4 u_base_color;\n"
    "uniform float u_metallic, u_roughness, u_normal_scale, u_occ_strength;\n"
    "uniform vec3 u_emissive;\n"
    "uniform sampler2D t_base, t_mr, t_normal, t_occ, t_emis;\n"
    "uniform int u_has_base, u_has_mr, u_has_normal, u_has_occ, u_has_emis;\n"
    "uniform int u_uv_base, u_uv_mr, u_uv_normal, u_uv_occ, u_uv_emis;\n"
    "uniform int u_alpha_mode; uniform float u_alpha_cutoff; uniform int u_unlit; uniform int u_has_tangent;\n"
    "uniform vec3 u_cam_pos;\n"
    "uniform vec3 u_light_dir[3]; uniform vec3 u_light_col[3];\n"
    "uniform vec3 u_sky, u_horizon, u_ground;\n"
    "uniform float u_exposure;\n"
    "const float PI = 3.14159265;\n"
    "vec2 uvsel(int s) { return s == 1 ? v_uv1 : v_uv0; }\n"
    "vec3 env(vec3 d, float rough) {\n"
    "  float y = d.y;\n"
    "  vec3 sharp = y > 0.0 ? mix(u_horizon, u_sky, pow(y, 0.6)) : mix(u_horizon, u_ground, pow(-y, 0.4));\n"
    "  // soft overhead and side 'softbox' panels give metals something to reflect\n"
    "  float box1 = smoothstep(0.86 - rough * 0.5, 0.97, dot(d, normalize(vec3(-0.4, 0.8, 0.45))));\n"
    "  float box2 = smoothstep(0.90 - rough * 0.5, 0.985, dot(d, normalize(vec3(0.85, 0.25, -0.3))));\n"
    "  sharp += vec3(2.2) * box1 * (1.0 - rough * 0.7) + vec3(1.0, 0.95, 0.9) * box2 * (1.0 - rough * 0.8);\n"
    "  vec3 blurred = mix(u_ground, u_sky, 0.5 + 0.5 * y) * 0.9 + 0.25;\n"
    "  return mix(sharp, blurred, rough * rough);\n"
    "}\n"
    "vec2 env_brdf(float rough, float nv) {\n"
    "  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);\n"
    "  const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);\n"
    "  vec4 r = rough * c0 + c1;\n"
    "  float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;\n"
    "  return vec2(-1.04, 1.04) * a004 + r.zw;\n"
    "}\n"
    "vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }\n"
    "void main() {\n"
    "  vec4 base = u_base_color * v_col;\n"
    "  if (u_has_base == 1) base *= texture(t_base, uvsel(u_uv_base));\n"
    "  if (u_alpha_mode == 1 && base.a < u_alpha_cutoff) discard;\n"
    "  if (u_unlit == 1) {\n"
    "    o_color = vec4(pow(clamp(base.rgb, 0.0, 1.0), vec3(1.0 / 2.2)), u_alpha_mode == 2 ? base.a : 1.0);\n"
    "    return;\n"
    "  }\n"
    "  float metal = u_metallic, rough = u_roughness;\n"
    "  if (u_has_mr == 1) { vec4 mr = texture(t_mr, uvsel(u_uv_mr)); rough *= mr.g; metal *= mr.b; }\n"
    "  rough = clamp(rough, 0.04, 1.0); metal = clamp(metal, 0.0, 1.0);\n"
    "  vec3 ng = normalize(v_nrm);\n"
    "  if (!gl_FrontFacing) ng = -ng;\n"
    "  vec3 n = ng;\n"
    "  if (u_has_normal == 1) {\n"
    "    vec2 uv = uvsel(u_uv_normal);\n"
    "    vec3 t, b;\n"
    "    if (u_has_tangent == 1) {\n"
    "      t = normalize(v_tan.xyz - ng * dot(ng, v_tan.xyz));\n"
    "      b = cross(ng, t) * v_tan.w;\n"
    "      if (!gl_FrontFacing) { t = -t; b = -b; }\n"
    "    } else {\n"
    "      vec3 uvdx = dFdx(vec3(uv, 0.0)), uvdy = dFdy(vec3(uv, 0.0));\n"
    "      vec3 t_ = (uvdy.t * dFdx(v_pos) - uvdx.t * dFdy(v_pos)) / (uvdx.s * uvdy.t - uvdy.s * uvdx.t);\n"
    "      t = normalize(t_ - ng * dot(ng, t_));\n"
    "      b = cross(ng, t);\n"
    "    }\n"
    "    vec3 tn = texture(t_normal, uv).xyz * 2.0 - 1.0;\n"
    "    tn.xy *= u_normal_scale;\n"
    "    vec3 pn = mat3(t, b, ng) * tn;\n"
    "    if (all(equal(pn, pn)) && dot(pn, pn) > 1e-8) n = normalize(pn);\n"
    "  }\n"
    "  vec3 v = normalize(u_cam_pos - v_pos);\n"
    "  float nv = clamp(abs(dot(n, v)), 0.001, 1.0);\n"
    "  vec3 diffuse_col = base.rgb * (1.0 - metal);\n"
    "  vec3 f0 = mix(vec3(0.04), base.rgb, metal);\n"
    "  float a = rough * rough;\n"
    "  vec3 col = vec3(0.0);\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    vec3 l = u_light_dir[i];\n"
    "    vec3 h = normalize(l + v);\n"
    "    float nl = clamp(dot(n, l), 0.0, 1.0);\n"
    "    if (nl <= 0.0) continue;\n"
    "    float nh = clamp(dot(n, h), 0.0, 1.0), vh = clamp(dot(v, h), 0.0, 1.0);\n"
    "    vec3 F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);\n"
    "    float a2 = a * a;\n"
    "    float d = nh * nh * (a2 - 1.0) + 1.0;\n"
    "    float D = a2 / (PI * d * d);\n"
    "    float gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);\n"
    "    float gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);\n"
    "    float V = 0.5 / max(gv + gl, 1e-5);\n"
    "    vec3 spec = F * D * V;\n"
    "    vec3 kd = (1.0 - F) * diffuse_col / PI;\n"
    "    col += (kd + spec) * u_light_col[i] * nl;\n"
    "  }\n"
    "  vec3 irr = mix(u_ground, u_sky, 0.5 + 0.5 * n.y) + 0.15;\n"
    "  vec2 ab = env_brdf(rough, nv);\n"
    "  vec3 r = reflect(-v, n);\n"
    "  vec3 amb = diffuse_col * irr + env(r, rough) * (f0 * ab.x + ab.y);\n"
    "  float ao = 1.0;\n"
    "  if (u_has_occ == 1) ao = 1.0 + u_occ_strength * (texture(t_occ, uvsel(u_uv_occ)).r - 1.0);\n"
    "  col += amb * ao;\n"
    "  vec3 em = u_emissive;\n"
    "  if (u_has_emis == 1) em *= texture(t_emis, uvsel(u_uv_emis)).rgb;\n"
    "  col += em;\n"
    "  col = aces(col * u_exposure);\n"
    "  o_color = vec4(pow(col, vec3(1.0 / 2.2)), u_alpha_mode == 2 ? base.a : 1.0);\n"
    "}\n";

static const char *BG_VS =
    "#version 330 core\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  v_uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static const char *BG_FS =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color;\n"
    "uniform vec3 u_top, u_bottom;\n"
    "void main() {\n"
    "  vec2 c = v_uv - vec2(0.5, 0.55);\n"
    "  float vig = 1.0 - 0.35 * dot(c, c);\n"
    "  o_color = vec4(mix(u_bottom, u_top, v_uv.y) * vig, 1.0);\n"
    "}\n";

static const char *GRID_VS =
    "#version 330 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "uniform mat4 u_viewproj; uniform vec3 u_center; uniform float u_extent;\n"
    "out vec3 v_world;\n"
    "void main() {\n"
    "  v_world = vec3(u_center.x + a_pos.x * u_extent, u_center.y, u_center.z + a_pos.y * u_extent);\n"
    "  gl_Position = u_viewproj * vec4(v_world, 1.0);\n"
    "}\n";

static const char *GRID_FS =
    "#version 330 core\n"
    "in vec3 v_world; out vec4 o_color;\n"
    "uniform vec3 u_center; uniform float u_cell; uniform float u_radius; uniform vec2 u_shadow;\n"
    "uniform int u_show_grid; uniform vec3 u_line_col;\n"
    "float grid(vec2 p, float cell) {\n"
    "  vec2 q = p / cell;\n"
    "  vec2 g = abs(fract(q - 0.5) - 0.5) / max(fwidth(q), vec2(1e-6));\n"
    "  return 1.0 - min(min(g.x, g.y), 1.0);\n"
    "}\n"
    "void main() {\n"
    "  vec2 p = v_world.xz;\n"
    "  float dist = length(p - u_center.xz);\n"
    "  float fade = 1.0 - smoothstep(u_radius * 1.2, u_radius * 4.0, dist);\n"
    "  float lines = 0.0;\n"
    "  if (u_show_grid == 1) lines = max(grid(p, u_cell) * 0.22, grid(p, u_cell * 10.0) * 0.45) * fade;\n"
    "  vec2 sd = (p - u_center.xz) / max(u_shadow, vec2(1e-5));\n"
    "  float shadow = (1.0 - smoothstep(0.2, 1.35, length(sd))) * 0.45;\n"
    "  float a = clamp(lines + shadow, 0.0, 1.0);\n"
    "  if (a <= 0.001) discard;\n"
    "  o_color = vec4(u_line_col * (lines / max(lines + shadow, 1e-5)), a);\n"
    "}\n";

static const char *HUD_VS =
    "#version 330 core\n"
    "layout(location=0) in vec2 a_pos; layout(location=1) in vec4 a_col;\n"
    "uniform vec2 u_size; out vec4 v_col;\n"
    "void main() { v_col = a_col; gl_Position = vec4(a_pos.x / u_size.x * 2.0 - 1.0, 1.0 - a_pos.y / u_size.y * 2.0, 0.0, 1.0); }\n";

static const char *HUD_FS =
    "#version 330 core\n"
    "in vec4 v_col; out vec4 o_color;\n"
    "void main() { o_color = v_col; }\n";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint link_program(const char *vs, const char *fs) {
    GLuint p = glCreateProgram();
    GLuint a = compile_shader(GL_VERTEX_SHADER, vs), b = compile_shader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "program link error:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(a);
    glDeleteShader(b);
    return p;
}

// ---------------------------------------------------------------------------------------------
// Model

enum { TEX_BASE, TEX_MR, TEX_NORMAL, TEX_OCC, TEX_EMIS, TEX_COUNT };
enum { ALPHA_OPAQUE, ALPHA_MASK, ALPHA_BLEND };

typedef struct {
    float base[4];
    float metallic, roughness, normal_scale, occ_strength;
    float emissive[3];
    int tex[TEX_COUNT]; // cgltf texture index or -1
    int uv[TEX_COUNT];
    int alpha_mode;
    float cutoff;
    int double_sided, unlit;
} Material;

typedef struct {
    GLuint vao, vbo[8], ebo;
    int nvbo;
    GLsizei count;
    GLenum mode;
    int indexed;
    int material;
    int has_skin_attrs, has_tangent;
    float bmin[3], bmax[3];
    // CPU copies kept only for skinned primitives, used to compute bind-pose bounds.
    float *cpu_pos, *cpu_joints, *cpu_weights;
    size_t nverts;
} Prim;

typedef struct {
    int first, count;
} MeshRef;

typedef struct {
    float t[3], r[4], s[3];
    float rest_t[3], rest_r[4], rest_s[3];
    int use_matrix;
    float matrix[16];
    mat4 world;
    int mesh, skin;
    int nchildren;
    int *children;
} Node;

typedef struct {
    int njoints;
    int *joints;
    mat4 *ibm;
    float *buf;
    GLuint tbo, tex;
} Skin;

typedef struct {
    float *in, *out;
    int n, comps, interp;
} Sampler;

typedef struct {
    int node, path, sampler;
} Channel;

typedef struct {
    char name[96];
    Sampler *samplers;
    int nsamplers;
    Channel *channels;
    int nchannels;
    float duration;
} Anim;

enum { JOB_UNUSED, JOB_PENDING, JOB_READY, JOB_DONE, JOB_FAILED };

typedef struct {
    SDL_AtomicInt state;
    int srgb, lin; // which GL variants the materials need
    unsigned char *pixels;
    int w, h;
} TexJob;

typedef struct {
    cgltf_data *data;
    char path[4096];
    char dir[4096];
    Prim *prims;
    int nprims;
    MeshRef *meshes;
    int nmeshes;
    Material *mats;
    int nmats;
    GLuint *tex_lin, *tex_srgb;
    int ntex;
    TexJob *jobs; // one per texture, decoded on worker threads
    SDL_Thread *workers[16];
    int nworkers;
    SDL_AtomicInt next_job, cancel;
    int tex_needed, tex_finished;
    Node *nodes;
    int nnodes;
    int *roots;
    int nroots;
    Skin *skins;
    int nskins;
    Anim *anims;
    int nanims;
    vec3 bmin, bmax;
    long tris;
    char warning[256];
} Model;

typedef struct {
    int prim;
    int node;
    int skinned;
    float depth;
} DrawItem;

// ---------------------------------------------------------------------------------------------
// App state

static struct {
    SDL_Window *win;
    int fb_w, fb_h, win_w, win_h;
    float density;
    GLuint mesh_prog, bg_prog, grid_prog, hud_prog;
    GLuint empty_vao, grid_vao, grid_vbo, hud_vao, hud_vbo;
    GLuint white_tex, dummy_tbo, dummy_tbo_tex;
    Model *model;
    char error[512];
    char loading[256]; // file name shown while the first model loads
    // camera
    float yaw, pitch, dist;
    vec3 target;
    float frame_dist;
    // toggles
    int show_grid, wireframe, auto_rotate, show_help, playing;
    int anim;
    double anim_time;
    int fullscreen;
    DrawItem *items;
    int nitems, cap_items;
} G;

static struct {
    GLint model, nmat, viewproj, skinned, joint_tex;
    GLint base_color, metallic, roughness, normal_scale, occ_strength, emissive;
    GLint t[TEX_COUNT], has[TEX_COUNT], uvs[TEX_COUNT];
    GLint alpha_mode, alpha_cutoff, unlit, has_tangent, cam_pos, light_dir, light_col;
    GLint sky, horizon, ground, exposure;
} U;

// ---------------------------------------------------------------------------------------------
// Loading helpers

static float *unpack_floats(const cgltf_accessor *a, int comps) {
    size_t n = a->count * (size_t)comps;
    float *buf = calloc(n ? n : 1, sizeof(float));
    if (cgltf_num_components(a->type) == (cgltf_size)comps) cgltf_accessor_unpack_floats(a, buf, n);
    return buf;
}

static int b64_val(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static unsigned char *b64_decode(const char *s, size_t *out_len) {
    size_t len = strlen(s);
    unsigned char *out = malloc(len / 4 * 3 + 4);
    size_t o = 0;
    unsigned buf = 0;
    int bits = 0;
    for (size_t i = 0; i < len; i++) {
        int v = b64_val((unsigned char)s[i]);
        if (v < 0) continue;
        buf = (buf << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = (unsigned char)((buf >> bits) & 0xff);
        }
    }
    *out_len = o;
    return out;
}

static unsigned char *load_image_pixels(Model *m, const cgltf_image *img, int *w, int *h) {
    int comp;
    if (img->buffer_view) {
        const uint8_t *p = cgltf_buffer_view_data(img->buffer_view);
        if (!p) return NULL;
        return stbi_load_from_memory(p, (int)img->buffer_view->size, w, h, &comp, 4);
    }
    if (!img->uri) return NULL;
    if (strncmp(img->uri, "data:", 5) == 0) {
        const char *comma = strchr(img->uri, ',');
        if (!comma) return NULL;
        size_t n;
        unsigned char *raw = b64_decode(comma + 1, &n);
        unsigned char *px = stbi_load_from_memory(raw, (int)n, w, h, &comp, 4);
        free(raw);
        return px;
    }
    char path[8192];
    snprintf(path, sizeof path, "%s%s", m->dir, img->uri);
    cgltf_decode_uri(path + strlen(m->dir));
    return stbi_load(path, w, h, &comp, 4);
}

// Decoding large PNG/JPEG textures is by far the slowest part of loading, so it runs on
// worker threads in parallel. The model shows up right away without textures; each texture
// is uploaded on the main thread as soon as its pixels are ready.
static int SDLCALL texture_worker(void *userdata) {
    Model *m = userdata;
    while (!SDL_GetAtomicInt(&m->cancel)) {
        int i = SDL_AddAtomicInt(&m->next_job, 1);
        if (i >= m->ntex) break;
        TexJob *j = &m->jobs[i];
        if (SDL_GetAtomicInt(&j->state) != JOB_PENDING) continue;
        j->pixels = load_image_pixels(m, m->data->textures[i].image, &j->w, &j->h);
        if (!j->pixels) {
            const cgltf_image *img = m->data->textures[i].image;
            fprintf(stderr, "warning: could not load image %s\n", img->uri ? img->uri : "(embedded)");
        }
        SDL_SetAtomicInt(&j->state, j->pixels ? JOB_READY : JOB_FAILED);
    }
    return 0;
}

static GLuint create_texture(const cgltf_texture *t, const TexJob *j, int srgb) {
    GLuint id;
    glGenTextures(1, &id);
    glActiveTexture(GL_TEXTURE0 + 8); // scratch unit so material bindings on 0..7 stay intact
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, j->w, j->h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 j->pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    GLint wrap_s = GL_REPEAT, wrap_t = GL_REPEAT, minf = GL_LINEAR_MIPMAP_LINEAR, magf = GL_LINEAR;
    if (t->sampler) {
        if (t->sampler->wrap_s) wrap_s = t->sampler->wrap_s;
        if (t->sampler->wrap_t) wrap_t = t->sampler->wrap_t;
        if (t->sampler->min_filter) minf = t->sampler->min_filter;
        if (t->sampler->mag_filter) magf = t->sampler->mag_filter;
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magf);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);
    while (glGetError() != GL_NO_ERROR) {
    }
    return id;
}

static void start_texture_jobs(Model *m) {
    m->jobs = calloc(m->ntex ? m->ntex : 1, sizeof(TexJob));
    for (int i = 0; i < m->nmats; i++)
        for (int k = 0; k < TEX_COUNT; k++) {
            int ti = m->mats[i].tex[k];
            if (ti < 0 || ti >= m->ntex) continue;
            if (k == TEX_BASE || k == TEX_EMIS) m->jobs[ti].srgb = 1;
            else m->jobs[ti].lin = 1;
        }
    for (int i = 0; i < m->ntex; i++) {
        TexJob *j = &m->jobs[i];
        if (!j->srgb && !j->lin) continue;
        const cgltf_texture *t = &m->data->textures[i];
        if (!t->image) {
            if (t->has_basisu || t->has_webp)
                snprintf(m->warning, sizeof m->warning, "Some textures use unsupported formats (KTX2/WebP)");
            continue;
        }
        SDL_SetAtomicInt(&j->state, JOB_PENDING);
        m->tex_needed++;
    }
    int n = SDL_min(SDL_GetNumLogicalCPUCores(), (int)SDL_arraysize(m->workers));
    n = SDL_min(n, m->tex_needed);
    for (int i = 0; i < n; i++) {
        m->workers[m->nworkers] = SDL_CreateThread(texture_worker, "texture", m);
        if (m->workers[m->nworkers]) m->nworkers++;
    }
    if (m->tex_needed > 0 && m->nworkers == 0) texture_worker(m); // no threads: decode synchronously
}

// Upload decoded textures to the GPU, spending at most budget_ms per call (but always at least
// one texture). Returns 1 while textures are still loading.
static int pump_textures(Model *m, double budget_ms) {
    Uint64 start = SDL_GetTicksNS();
    for (int i = 0; i < m->ntex && m->tex_finished < m->tex_needed; i++) {
        TexJob *j = &m->jobs[i];
        int st = SDL_GetAtomicInt(&j->state);
        if (st == JOB_FAILED) {
            SDL_SetAtomicInt(&j->state, JOB_DONE);
            m->tex_finished++;
        } else if (st == JOB_READY) {
            if (SDL_GetTicksNS() - start > (Uint64)(budget_ms * 1e6)) break;
            const cgltf_texture *t = &m->data->textures[i];
            if (j->srgb) m->tex_srgb[i] = create_texture(t, j, 1);
            if (j->lin) m->tex_lin[i] = create_texture(t, j, 0);
            stbi_image_free(j->pixels);
            j->pixels = NULL;
            SDL_SetAtomicInt(&j->state, JOB_DONE);
            m->tex_finished++;
        }
    }
    return m->tex_finished < m->tex_needed;
}

static GLuint get_texture(Model *m, int ti, int srgb) {
    if (ti < 0 || ti >= m->ntex) return 0;
    return srgb ? m->tex_srgb[ti] : m->tex_lin[ti];
}

static int tex_index(const Model *m, const cgltf_texture_view *v) {
    return v->texture ? (int)(v->texture - m->data->textures) : -1;
}

static void load_materials(Model *m) {
    cgltf_data *d = m->data;
    m->nmats = (int)d->materials_count;
    m->mats = calloc(m->nmats ? m->nmats : 1, sizeof(Material));
    for (int i = 0; i < m->nmats; i++) {
        const cgltf_material *cm = &d->materials[i];
        Material *mt = &m->mats[i];
        for (int k = 0; k < TEX_COUNT; k++) mt->tex[k] = -1;
        mt->base[0] = mt->base[1] = mt->base[2] = mt->base[3] = 1;
        mt->metallic = 1;
        mt->roughness = 1;
        if (cm->has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness *p = &cm->pbr_metallic_roughness;
            memcpy(mt->base, p->base_color_factor, sizeof mt->base);
            mt->metallic = p->metallic_factor;
            mt->roughness = p->roughness_factor;
            mt->tex[TEX_BASE] = tex_index(m, &p->base_color_texture);
            mt->uv[TEX_BASE] = p->base_color_texture.texcoord;
            mt->tex[TEX_MR] = tex_index(m, &p->metallic_roughness_texture);
            mt->uv[TEX_MR] = p->metallic_roughness_texture.texcoord;
        } else if (cm->has_pbr_specular_glossiness) {
            const cgltf_pbr_specular_glossiness *p = &cm->pbr_specular_glossiness;
            memcpy(mt->base, p->diffuse_factor, sizeof mt->base);
            mt->metallic = 0;
            mt->roughness = 1.0f - p->glossiness_factor;
            mt->tex[TEX_BASE] = tex_index(m, &p->diffuse_texture);
            mt->uv[TEX_BASE] = p->diffuse_texture.texcoord;
        }
        mt->tex[TEX_NORMAL] = tex_index(m, &cm->normal_texture);
        mt->uv[TEX_NORMAL] = cm->normal_texture.texcoord;
        mt->normal_scale = cm->normal_texture.texture ? cm->normal_texture.scale : 1.0f;
        mt->tex[TEX_OCC] = tex_index(m, &cm->occlusion_texture);
        mt->uv[TEX_OCC] = cm->occlusion_texture.texcoord;
        mt->occ_strength = cm->occlusion_texture.texture ? cm->occlusion_texture.scale : 1.0f;
        mt->tex[TEX_EMIS] = tex_index(m, &cm->emissive_texture);
        mt->uv[TEX_EMIS] = cm->emissive_texture.texcoord;
        float es = cm->has_emissive_strength ? cm->emissive_strength.emissive_strength : 1.0f;
        for (int k = 0; k < 3; k++) mt->emissive[k] = cm->emissive_factor[k] * es;
        mt->alpha_mode = cm->alpha_mode == cgltf_alpha_mode_mask    ? ALPHA_MASK
                         : cm->alpha_mode == cgltf_alpha_mode_blend ? ALPHA_BLEND
                                                                    : ALPHA_OPAQUE;
        mt->cutoff = cm->alpha_cutoff;
        mt->double_sided = cm->double_sided;
        mt->unlit = cm->unlit;
    }
}

static GLenum gl_mode(cgltf_primitive_type t) {
    switch (t) {
    case cgltf_primitive_type_points: return GL_POINTS;
    case cgltf_primitive_type_lines: return GL_LINES;
    case cgltf_primitive_type_line_loop: return GL_LINE_LOOP;
    case cgltf_primitive_type_line_strip: return GL_LINE_STRIP;
    case cgltf_primitive_type_triangle_strip: return GL_TRIANGLE_STRIP;
    case cgltf_primitive_type_triangle_fan: return GL_TRIANGLE_FAN;
    default: return GL_TRIANGLES;
    }
}

static void upload_attr(Prim *p, int loc, const float *data, size_t nverts, int comps) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nverts * comps * sizeof(float)), data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, comps, GL_FLOAT, GL_FALSE, 0, 0);
    p->vbo[p->nvbo++] = vbo;
}

static void compute_normals(float *nrm, const float *pos, size_t nverts, const uint32_t *idx, size_t nidx) {
    memset(nrm, 0, nverts * 3 * sizeof(float));
    size_t ntri = (idx ? nidx : nverts) / 3;
    for (size_t t = 0; t < ntri; t++) {
        uint32_t i0 = idx ? idx[t * 3] : (uint32_t)(t * 3);
        uint32_t i1 = idx ? idx[t * 3 + 1] : (uint32_t)(t * 3 + 1);
        uint32_t i2 = idx ? idx[t * 3 + 2] : (uint32_t)(t * 3 + 2);
        if (i0 >= nverts || i1 >= nverts || i2 >= nverts) continue;
        vec3 a = v3(pos[i0 * 3], pos[i0 * 3 + 1], pos[i0 * 3 + 2]);
        vec3 b = v3(pos[i1 * 3], pos[i1 * 3 + 1], pos[i1 * 3 + 2]);
        vec3 c = v3(pos[i2 * 3], pos[i2 * 3 + 1], pos[i2 * 3 + 2]);
        vec3 n = v3_cross(v3_sub(b, a), v3_sub(c, a));
        uint32_t ids[3] = {i0, i1, i2};
        for (int k = 0; k < 3; k++) {
            nrm[ids[k] * 3] += n.x;
            nrm[ids[k] * 3 + 1] += n.y;
            nrm[ids[k] * 3 + 2] += n.z;
        }
    }
    for (size_t i = 0; i < nverts; i++) {
        vec3 n = v3_norm(v3(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]));
        nrm[i * 3] = n.x;
        nrm[i * 3 + 1] = n.y;
        nrm[i * 3 + 2] = n.z;
    }
}

static int load_prim(Model *m, const cgltf_primitive *cp, Prim *p) {
    const cgltf_accessor *apos = NULL, *anrm = NULL, *auv0 = NULL, *auv1 = NULL, *acol = NULL;
    const cgltf_accessor *ajoints = NULL, *aweights = NULL, *atan = NULL;
    for (size_t i = 0; i < cp->attributes_count; i++) {
        const cgltf_attribute *at = &cp->attributes[i];
        switch (at->type) {
        case cgltf_attribute_type_position: apos = at->data; break;
        case cgltf_attribute_type_normal: anrm = at->data; break;
        case cgltf_attribute_type_tangent: atan = at->data; break;
        case cgltf_attribute_type_texcoord:
            if (at->index == 0) auv0 = at->data;
            else if (at->index == 1) auv1 = at->data;
            break;
        case cgltf_attribute_type_color:
            if (at->index == 0) acol = at->data;
            break;
        case cgltf_attribute_type_joints:
            if (at->index == 0) ajoints = at->data;
            break;
        case cgltf_attribute_type_weights:
            if (at->index == 0) aweights = at->data;
            break;
        default: break;
        }
    }
    if (cp->has_draco_mesh_compression) {
        snprintf(m->warning, sizeof m->warning, "Draco-compressed meshes are not supported");
        return 0;
    }
    if (!apos || apos->count == 0) return 0;

    memset(p, 0, sizeof *p);
    p->mode = gl_mode(cp->type);
    p->material = cp->material ? (int)(cp->material - m->data->materials) : -1;
    size_t nv = apos->count;
    p->nverts = nv;

    glGenVertexArrays(1, &p->vao);
    glBindVertexArray(p->vao);

    float *pos = unpack_floats(apos, 3);
    for (int k = 0; k < 3; k++) {
        p->bmin[k] = INFINITY;
        p->bmax[k] = -INFINITY;
    }
    for (size_t i = 0; i < nv; i++)
        for (int k = 0; k < 3; k++) {
            float v = pos[i * 3 + k];
            if (v < p->bmin[k]) p->bmin[k] = v;
            if (v > p->bmax[k]) p->bmax[k] = v;
        }
    upload_attr(p, 0, pos, nv, 3);

    uint32_t *idx = NULL;
    size_t nidx = 0;
    if (cp->indices) {
        nidx = cp->indices->count;
        idx = malloc((nidx ? nidx : 1) * sizeof(uint32_t));
        for (size_t i = 0; i < nidx; i++) idx[i] = (uint32_t)cgltf_accessor_read_index(cp->indices, i);
        glGenBuffers(1, &p->ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, p->ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(nidx * sizeof(uint32_t)), idx, GL_STATIC_DRAW);
        p->indexed = 1;
        p->count = (GLsizei)nidx;
    } else {
        p->count = (GLsizei)nv;
    }

    if (anrm && anrm->count == nv) {
        float *n = unpack_floats(anrm, 3);
        upload_attr(p, 1, n, nv, 3);
        free(n);
    } else if (p->mode == GL_TRIANGLES) {
        float *n = malloc(nv * 3 * sizeof(float));
        compute_normals(n, pos, nv, idx, nidx);
        upload_attr(p, 1, n, nv, 3);
        free(n);
    }
    if (auv0 && auv0->count == nv) {
        float *b = unpack_floats(auv0, 2);
        upload_attr(p, 2, b, nv, 2);
        free(b);
    }
    if (auv1 && auv1->count == nv) {
        float *b = unpack_floats(auv1, 2);
        upload_attr(p, 3, b, nv, 2);
        free(b);
    }
    if (acol && acol->count == nv) {
        int c = (int)cgltf_num_components(acol->type);
        float *b = unpack_floats(acol, c);
        if (c == 3) {
            float *b4 = malloc(nv * 4 * sizeof(float));
            for (size_t i = 0; i < nv; i++) {
                b4[i * 4] = b[i * 3];
                b4[i * 4 + 1] = b[i * 3 + 1];
                b4[i * 4 + 2] = b[i * 3 + 2];
                b4[i * 4 + 3] = 1;
            }
            free(b);
            b = b4;
        }
        upload_attr(p, 4, b, nv, 4);
        free(b);
    }
    if (ajoints && aweights && ajoints->count == nv && aweights->count == nv) {
        float *j = unpack_floats(ajoints, 4);
        float *w = unpack_floats(aweights, 4);
        upload_attr(p, 5, j, nv, 4);
        upload_attr(p, 6, w, nv, 4);
        p->has_skin_attrs = 1;
        p->cpu_joints = j;
        p->cpu_weights = w;
        p->cpu_pos = pos;
        pos = NULL;
    }
    if (atan && atan->count == nv && cgltf_num_components(atan->type) == 4) {
        float *t = unpack_floats(atan, 4);
        upload_attr(p, 7, t, nv, 4);
        free(t);
        p->has_tangent = 1;
    }
    free(pos);
    free(idx);
    glBindVertexArray(0);

    if (p->mode == GL_TRIANGLES) m->tris += p->count / 3;
    else if (p->mode == GL_TRIANGLE_STRIP || p->mode == GL_TRIANGLE_FAN) m->tris += p->count > 2 ? p->count - 2 : 0;
    return 1;
}

static void load_nodes(Model *m) {
    cgltf_data *d = m->data;
    m->nnodes = (int)d->nodes_count;
    m->nodes = calloc(m->nnodes ? m->nnodes : 1, sizeof(Node));
    for (int i = 0; i < m->nnodes; i++) {
        const cgltf_node *cn = &d->nodes[i];
        Node *n = &m->nodes[i];
        n->t[0] = n->t[1] = n->t[2] = 0;
        n->r[0] = n->r[1] = n->r[2] = 0;
        n->r[3] = 1;
        n->s[0] = n->s[1] = n->s[2] = 1;
        if (cn->has_translation) memcpy(n->t, cn->translation, sizeof n->t);
        if (cn->has_rotation) memcpy(n->r, cn->rotation, sizeof n->r);
        if (cn->has_scale) memcpy(n->s, cn->scale, sizeof n->s);
        if (cn->has_matrix) {
            n->use_matrix = 1;
            memcpy(n->matrix, cn->matrix, sizeof n->matrix);
        }
        memcpy(n->rest_t, n->t, sizeof n->t);
        memcpy(n->rest_r, n->r, sizeof n->r);
        memcpy(n->rest_s, n->s, sizeof n->s);
        n->mesh = cn->mesh ? (int)(cn->mesh - d->meshes) : -1;
        n->skin = cn->skin ? (int)(cn->skin - d->skins) : -1;
        n->nchildren = (int)cn->children_count;
        n->children = malloc((n->nchildren ? n->nchildren : 1) * sizeof(int));
        for (int c = 0; c < n->nchildren; c++) n->children[c] = (int)(cn->children[c] - d->nodes);
        n->world = m4_identity();
    }
    const cgltf_scene *scene = d->scene ? d->scene : (d->scenes_count ? &d->scenes[0] : NULL);
    if (scene) {
        m->nroots = (int)scene->nodes_count;
        m->roots = malloc((m->nroots ? m->nroots : 1) * sizeof(int));
        for (int i = 0; i < m->nroots; i++) m->roots[i] = (int)(scene->nodes[i] - d->nodes);
    } else {
        m->roots = malloc((m->nnodes ? m->nnodes : 1) * sizeof(int));
        for (int i = 0; i < m->nnodes; i++)
            if (!d->nodes[i].parent) m->roots[m->nroots++] = i;
    }
}

static void load_skins(Model *m) {
    cgltf_data *d = m->data;
    m->nskins = (int)d->skins_count;
    m->skins = calloc(m->nskins ? m->nskins : 1, sizeof(Skin));
    for (int i = 0; i < m->nskins; i++) {
        const cgltf_skin *cs = &d->skins[i];
        Skin *s = &m->skins[i];
        s->njoints = (int)cs->joints_count;
        s->joints = malloc((s->njoints ? s->njoints : 1) * sizeof(int));
        s->ibm = malloc((s->njoints ? s->njoints : 1) * sizeof(mat4));
        s->buf = malloc((s->njoints ? s->njoints : 1) * sizeof(mat4));
        for (int j = 0; j < s->njoints; j++) {
            s->joints[j] = (int)(cs->joints[j] - d->nodes);
            s->ibm[j] = m4_identity();
            if (cs->inverse_bind_matrices) cgltf_accessor_read_float(cs->inverse_bind_matrices, j, s->ibm[j].m, 16);
        }
        glGenBuffers(1, &s->tbo);
        glBindBuffer(GL_TEXTURE_BUFFER, s->tbo);
        glBufferData(GL_TEXTURE_BUFFER, (GLsizeiptr)((s->njoints ? s->njoints : 1) * sizeof(mat4)), NULL, GL_DYNAMIC_DRAW);
        glGenTextures(1, &s->tex);
        glBindTexture(GL_TEXTURE_BUFFER, s->tex);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, s->tbo);
    }
}

static void load_anims(Model *m) {
    cgltf_data *d = m->data;
    m->nanims = (int)d->animations_count;
    m->anims = calloc(m->nanims ? m->nanims : 1, sizeof(Anim));
    for (int i = 0; i < m->nanims; i++) {
        const cgltf_animation *ca = &d->animations[i];
        Anim *a = &m->anims[i];
        if (ca->name && ca->name[0]) snprintf(a->name, sizeof a->name, "%s", ca->name);
        else snprintf(a->name, sizeof a->name, "Animation %d", i + 1);
        a->nsamplers = (int)ca->samplers_count;
        a->samplers = calloc(a->nsamplers ? a->nsamplers : 1, sizeof(Sampler));
        for (int s = 0; s < a->nsamplers; s++) {
            const cgltf_animation_sampler *cs = &ca->samplers[s];
            Sampler *sp = &a->samplers[s];
            sp->n = (int)cs->input->count;
            sp->comps = (int)cgltf_num_components(cs->output->type);
            sp->interp = cs->interpolation;
            sp->in = unpack_floats(cs->input, 1);
            sp->out = unpack_floats(cs->output, sp->comps);
            if (sp->n > 0 && sp->in[sp->n - 1] > a->duration) a->duration = sp->in[sp->n - 1];
        }
        a->channels = calloc(ca->channels_count ? ca->channels_count : 1, sizeof(Channel));
        for (size_t c = 0; c < ca->channels_count; c++) {
            const cgltf_animation_channel *cc = &ca->channels[c];
            if (!cc->target_node || !cc->sampler) continue;
            if (cc->target_path != cgltf_animation_path_type_translation &&
                cc->target_path != cgltf_animation_path_type_rotation &&
                cc->target_path != cgltf_animation_path_type_scale)
                continue;
            Channel *ch = &a->channels[a->nchannels++];
            ch->node = (int)(cc->target_node - d->nodes);
            ch->path = cc->target_path;
            ch->sampler = (int)(cc->sampler - ca->samplers);
        }
    }
}

static void free_model(Model *m) {
    if (!m) return;
    SDL_SetAtomicInt(&m->cancel, 1);
    for (int i = 0; i < m->nworkers; i++) SDL_WaitThread(m->workers[i], NULL);
    for (int i = 0; m->jobs && i < m->ntex; i++) stbi_image_free(m->jobs[i].pixels);
    free(m->jobs);
    for (int i = 0; i < m->nprims; i++) {
        Prim *p = &m->prims[i];
        glDeleteVertexArrays(1, &p->vao);
        glDeleteBuffers(p->nvbo, p->vbo);
        if (p->ebo) glDeleteBuffers(1, &p->ebo);
        free(p->cpu_pos);
        free(p->cpu_joints);
        free(p->cpu_weights);
    }
    for (int i = 0; i < m->ntex; i++) {
        if (m->tex_lin[i]) glDeleteTextures(1, &m->tex_lin[i]);
        if (m->tex_srgb[i]) glDeleteTextures(1, &m->tex_srgb[i]);
    }
    for (int i = 0; i < m->nnodes; i++) free(m->nodes[i].children);
    for (int i = 0; i < m->nskins; i++) {
        Skin *s = &m->skins[i];
        glDeleteBuffers(1, &s->tbo);
        glDeleteTextures(1, &s->tex);
        free(s->joints);
        free(s->ibm);
        free(s->buf);
    }
    for (int i = 0; i < m->nanims; i++) {
        Anim *a = &m->anims[i];
        for (int s = 0; s < a->nsamplers; s++) {
            free(a->samplers[s].in);
            free(a->samplers[s].out);
        }
        free(a->samplers);
        free(a->channels);
    }
    free(m->prims);
    free(m->meshes);
    free(m->mats);
    free(m->tex_lin);
    free(m->tex_srgb);
    free(m->nodes);
    free(m->roots);
    free(m->skins);
    free(m->anims);
    if (m->data) cgltf_free(m->data);
    free(m);
}

// ---------------------------------------------------------------------------------------------
// Scene evaluation

static void update_world(Model *m, int ni, const mat4 *parent, int depth) {
    if (depth > 256) return;
    Node *n = &m->nodes[ni];
    mat4 local;
    if (n->use_matrix) memcpy(local.m, n->matrix, sizeof local.m);
    else local = m4_trs(n->t, n->r, n->s);
    n->world = parent ? m4_mul(*parent, local) : local;
    for (int c = 0; c < n->nchildren; c++) update_world(m, n->children[c], &n->world, depth + 1);
}

static void update_skins(Model *m) {
    for (int i = 0; i < m->nskins; i++) {
        Skin *s = &m->skins[i];
        mat4 *out = (mat4 *)s->buf;
        for (int j = 0; j < s->njoints; j++) out[j] = m4_mul(m->nodes[s->joints[j]].world, s->ibm[j]);
        glBindBuffer(GL_TEXTURE_BUFFER, s->tbo);
        glBufferSubData(GL_TEXTURE_BUFFER, 0, (GLsizeiptr)(s->njoints * sizeof(mat4)), s->buf);
    }
}

static void sample_channel(const Sampler *s, float t, int is_rot, float *out) {
    int c = s->comps, n = s->n;
    if (n == 0 || c > 4) return;
    int cubic = s->interp == cgltf_interpolation_type_cubic_spline;
    int stride = cubic ? 3 : 1, off = cubic ? 1 : 0;
    if (n == 1 || t <= s->in[0]) {
        memcpy(out, &s->out[off * c], c * sizeof(float));
        return;
    }
    if (t >= s->in[n - 1]) {
        memcpy(out, &s->out[((n - 1) * stride + off) * c], c * sizeof(float));
        return;
    }
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (s->in[mid] <= t) lo = mid;
        else hi = mid;
    }
    float t0 = s->in[lo], t1 = s->in[hi];
    float dt = t1 - t0;
    float u = dt > 0 ? (t - t0) / dt : 0;
    if (s->interp == cgltf_interpolation_type_step) {
        memcpy(out, &s->out[lo * c], c * sizeof(float));
    } else if (!cubic) {
        const float *a = &s->out[lo * c], *b = &s->out[hi * c];
        if (is_rot) quat_slerp(a, b, u, out);
        else
            for (int k = 0; k < c; k++) out[k] = a[k] + (b[k] - a[k]) * u;
    } else {
        const float *p0 = &s->out[(lo * 3 + 1) * c], *m0 = &s->out[(lo * 3 + 2) * c];
        const float *p1 = &s->out[(hi * 3 + 1) * c], *m1 = &s->out[(hi * 3 + 0) * c];
        float u2 = u * u, u3 = u2 * u;
        float h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
        float l = 0;
        for (int k = 0; k < c; k++) {
            out[k] = h00 * p0[k] + h10 * dt * m0[k] + h01 * p1[k] + h11 * dt * m1[k];
            l += out[k] * out[k];
        }
        if (is_rot && l > 0)
            for (int k = 0; k < c; k++) out[k] /= sqrtf(l);
    }
}

static void reset_pose(Model *m) {
    for (int i = 0; i < m->nnodes; i++) {
        Node *n = &m->nodes[i];
        memcpy(n->t, n->rest_t, sizeof n->t);
        memcpy(n->r, n->rest_r, sizeof n->r);
        memcpy(n->s, n->rest_s, sizeof n->s);
    }
}

static void apply_anim(Model *m, int ai, float t) {
    if (ai < 0 || ai >= m->nanims) return;
    Anim *a = &m->anims[ai];
    for (int i = 0; i < a->nchannels; i++) {
        Channel *ch = &a->channels[i];
        Sampler *s = &a->samplers[ch->sampler];
        Node *n = &m->nodes[ch->node];
        n->use_matrix = 0;
        if (ch->path == cgltf_animation_path_type_translation && s->comps == 3) sample_channel(s, t, 0, n->t);
        else if (ch->path == cgltf_animation_path_type_rotation && s->comps == 4) sample_channel(s, t, 1, n->r);
        else if (ch->path == cgltf_animation_path_type_scale && s->comps == 3) sample_channel(s, t, 0, n->s);
    }
}

static void evaluate_scene(Model *m) {
    for (int i = 0; i < m->nroots; i++) update_world(m, m->roots[i], NULL, 0);
}

static void expand_bounds(vec3 *bmin, vec3 *bmax, vec3 p) {
    if (p.x < bmin->x) bmin->x = p.x;
    if (p.y < bmin->y) bmin->y = p.y;
    if (p.z < bmin->z) bmin->z = p.z;
    if (p.x > bmax->x) bmax->x = p.x;
    if (p.y > bmax->y) bmax->y = p.y;
    if (p.z > bmax->z) bmax->z = p.z;
}

static void compute_bounds(Model *m) {
    m->bmin = v3(INFINITY, INFINITY, INFINITY);
    m->bmax = v3(-INFINITY, -INFINITY, -INFINITY);
    for (int ni = 0; ni < m->nnodes; ni++) {
        Node *n = &m->nodes[ni];
        if (n->mesh < 0) continue;
        MeshRef *mr = &m->meshes[n->mesh];
        for (int pi = mr->first; pi < mr->first + mr->count; pi++) {
            Prim *p = &m->prims[pi];
            if (n->skin >= 0 && p->has_skin_attrs && p->cpu_pos) {
                Skin *s = &m->skins[n->skin];
                mat4 *jm = (mat4 *)s->buf;
                for (int j = 0; j < s->njoints; j++) jm[j] = m4_mul(m->nodes[s->joints[j]].world, s->ibm[j]);
                for (size_t v = 0; v < p->nverts; v++) {
                    vec3 src = v3(p->cpu_pos[v * 3], p->cpu_pos[v * 3 + 1], p->cpu_pos[v * 3 + 2]);
                    vec3 acc = v3(0, 0, 0);
                    float wsum = 0;
                    for (int k = 0; k < 4; k++) {
                        float w = p->cpu_weights[v * 4 + k];
                        int j = (int)p->cpu_joints[v * 4 + k];
                        if (w <= 0 || j < 0 || j >= s->njoints) continue;
                        acc = v3_add(acc, v3_scale(m4_point(&jm[j], src), w));
                        wsum += w;
                    }
                    if (wsum > 0) expand_bounds(&m->bmin, &m->bmax, v3_scale(acc, 1.0f / wsum));
                }
            } else {
                for (int c = 0; c < 8; c++) {
                    vec3 corner = v3(c & 1 ? p->bmax[0] : p->bmin[0], c & 2 ? p->bmax[1] : p->bmin[1],
                                     c & 4 ? p->bmax[2] : p->bmin[2]);
                    // only nodes reachable from the scene have valid world matrices; that's fine
                    expand_bounds(&m->bmin, &m->bmax, m4_point(&n->world, corner));
                }
            }
        }
    }
    if (m->bmin.x > m->bmax.x) {
        m->bmin = v3(-1, -1, -1);
        m->bmax = v3(1, 1, 1);
    }
    for (int i = 0; i < m->nprims; i++) {
        Prim *p = &m->prims[i];
        free(p->cpu_pos);
        free(p->cpu_joints);
        free(p->cpu_weights);
        p->cpu_pos = p->cpu_joints = p->cpu_weights = NULL;
    }
}

static const char *basename_of(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

static Model *load_model(const char *path, char *err, size_t errlen) {
    cgltf_options opt = {0};
    cgltf_data *data = NULL;
    cgltf_result r = cgltf_parse_file(&opt, path, &data);
    if (r != cgltf_result_success) {
        snprintf(err, errlen, "Could not open %.200s (cgltf error %d)", basename_of(path), (int)r);
        return NULL;
    }
    r = cgltf_load_buffers(&opt, data, path);
    if (r != cgltf_result_success) {
        snprintf(err, errlen, "Could not load buffers for %.200s (missing .bin file?)", basename_of(path));
        cgltf_free(data);
        return NULL;
    }
    Model *m = calloc(1, sizeof(Model));
    m->data = data;
    snprintf(m->path, sizeof m->path, "%s", path);
    snprintf(m->dir, sizeof m->dir, "%s", path);
    char *slash = strrchr(m->dir, '/');
    if (slash) slash[1] = 0;
    else m->dir[0] = 0;

    for (size_t i = 0; i < data->extensions_required_count; i++) {
        const char *e = data->extensions_required[i];
        if (strcmp(e, "EXT_meshopt_compression") == 0 || strcmp(e, "KHR_draco_mesh_compression") == 0 ||
            strcmp(e, "KHR_texture_basisu") == 0 || strcmp(e, "EXT_texture_webp") == 0)
            snprintf(m->warning, sizeof m->warning, "Required extension not supported: %s", e);
    }

    m->ntex = (int)data->textures_count;
    m->tex_lin = calloc(m->ntex ? m->ntex : 1, sizeof(GLuint));
    m->tex_srgb = calloc(m->ntex ? m->ntex : 1, sizeof(GLuint));
    load_materials(m);
    start_texture_jobs(m);

    size_t total = 0;
    for (size_t i = 0; i < data->meshes_count; i++) total += data->meshes[i].primitives_count;
    m->prims = calloc(total ? total : 1, sizeof(Prim));
    m->nmeshes = (int)data->meshes_count;
    m->meshes = calloc(m->nmeshes ? m->nmeshes : 1, sizeof(MeshRef));
    for (size_t i = 0; i < data->meshes_count; i++) {
        m->meshes[i].first = m->nprims;
        for (size_t p = 0; p < data->meshes[i].primitives_count; p++)
            if (load_prim(m, &data->meshes[i].primitives[p], &m->prims[m->nprims])) m->nprims++;
        m->meshes[i].count = m->nprims - m->meshes[i].first;
    }
    load_nodes(m);
    load_skins(m);
    load_anims(m);
    evaluate_scene(m);
    compute_bounds(m);
    return m;
}

// ---------------------------------------------------------------------------------------------
// Camera

static void frame_model(void) {
    Model *m = G.model;
    if (!m) {
        G.target = v3(0, 0, 0);
        G.dist = 4;
        G.frame_dist = 4;
        return;
    }
    G.target = v3_scale(v3_add(m->bmin, m->bmax), 0.5f);
    float radius = v3_len(v3_sub(m->bmax, m->bmin)) * 0.5f;
    if (radius < 1e-6f) radius = 1;
    float aspect = G.fb_h > 0 ? (float)G.fb_w / (float)G.fb_h : 1.0f;
    float fov = aspect < 1 ? 2 * atanf(tanf(FOV_Y * 0.5f) * aspect) : FOV_Y;
    G.dist = radius / sinf(fov * 0.5f) * 1.05f;
    G.frame_dist = G.dist;
}

static void reset_view(void) {
    G.yaw = 0.6f;
    G.pitch = 0.3f;
    frame_model();
}

static vec3 camera_eye(void) {
    vec3 off = v3(cosf(G.pitch) * sinf(G.yaw), sinf(G.pitch), cosf(G.pitch) * cosf(G.yaw));
    return v3_add(G.target, v3_scale(off, G.dist));
}

// ---------------------------------------------------------------------------------------------
// HUD (stb_easy_font)

typedef struct {
    float x, y;
    unsigned char rgba[4];
} HudVert;

static HudVert *hud_verts;
static int hud_n, hud_cap;

static void hud_push(float x, float y, const unsigned char c[4]) {
    if (hud_n == hud_cap) {
        hud_cap = hud_cap ? hud_cap * 2 : 4096;
        hud_verts = realloc(hud_verts, hud_cap * sizeof(HudVert));
    }
    HudVert *v = &hud_verts[hud_n++];
    v->x = x;
    v->y = y;
    memcpy(v->rgba, c, 4);
}

static void hud_rect(float x, float y, float w, float h, const unsigned char c[4]) {
    hud_push(x, y, c);
    hud_push(x + w, y, c);
    hud_push(x + w, y + h, c);
    hud_push(x, y, c);
    hud_push(x + w, y + h, c);
    hud_push(x, y + h, c);
}

static float hud_scale(void) { return 2.0f * G.density; }

static void hud_text_raw(float x, float y, const char *text, const unsigned char c[4]) {
    static char quads[200000];
    float s = hud_scale();
    int nq = stb_easy_font_print(0, 0, (char *)text, NULL, quads, sizeof quads);
    for (int q = 0; q < nq; q++) {
        float *v[4];
        for (int k = 0; k < 4; k++) v[k] = (float *)(quads + (q * 4 + k) * 16);
        float px[4], py[4];
        for (int k = 0; k < 4; k++) {
            px[k] = x + v[k][0] * s;
            py[k] = y + v[k][1] * s;
        }
        hud_push(px[0], py[0], c);
        hud_push(px[1], py[1], c);
        hud_push(px[2], py[2], c);
        hud_push(px[0], py[0], c);
        hud_push(px[2], py[2], c);
        hud_push(px[3], py[3], c);
    }
}

static void hud_text(float x, float y, const char *text, const unsigned char c[4]) {
    static const unsigned char shadow[4] = {0, 0, 0, 160};
    float o = G.density;
    hud_text_raw(x + o, y + o, text, shadow);
    hud_text_raw(x, y, text, c);
}

static float hud_text_width(const char *text) { return stb_easy_font_width((char *)text) * hud_scale(); }
static float hud_line_height(void) { return 12.0f * hud_scale(); }

static const char *HELP_TEXT =
    "Left drag           Orbit\n"
    "Right / Middle drag Pan  (or Shift + Left drag)\n"
    "Scroll              Zoom\n"
    "F / Double-click    Frame model\n"
    "R                   Reset view\n"
    "A                   Auto-rotate\n"
    "G                   Toggle grid\n"
    "W                   Wireframe\n"
    "Space               Play / pause animation\n"
    "N / P               Next / previous animation\n"
    "O                   Open file...\n"
    "F11                 Fullscreen\n"
    "H                   Hide this help\n"
    "Q / Esc             Quit";

static void build_hud(void) {
    static const unsigned char white[4] = {235, 235, 240, 255};
    static const unsigned char dim[4] = {160, 165, 175, 255};
    static const unsigned char red[4] = {255, 110, 100, 255};
    static const unsigned char amber[4] = {255, 200, 90, 255};
    static const unsigned char panel[4] = {10, 12, 16, 200};
    hud_n = 0;
    float pad = 12 * G.density;
    float lh = hud_line_height();
    float y = pad;
    char buf[512];
    Model *m = G.model;
    if (m) {
        hud_text(pad, y, basename_of(m->path), white);
        y += lh;
        snprintf(buf, sizeof buf, "%ld triangles   %d meshes   %d materials   %d textures", m->tris, m->nmeshes,
                 m->nmats, m->ntex);
        hud_text(pad, y, buf, dim);
        y += lh;
        if (m->nanims > 0) {
            Anim *a = &m->anims[G.anim];
            snprintf(buf, sizeof buf, "Animation %d/%d: %s  %.2f / %.2fs%s", G.anim + 1, m->nanims, a->name,
                     a->duration > 0 ? fmod(G.anim_time, a->duration) : 0.0, a->duration,
                     G.playing ? "" : "  (paused)");
            hud_text(pad, y, buf, dim);
            y += lh;
        }
        if (m->tex_finished < m->tex_needed) {
            snprintf(buf, sizeof buf, "Loading textures %d/%d...", m->tex_finished, m->tex_needed);
            hud_text(pad, y, buf, amber);
            y += lh;
        }
        if (m->warning[0]) {
            hud_text(pad, y, m->warning, amber);
            y += lh;
        }
    } else if (!G.error[0]) {
        char loading[300];
        snprintf(loading, sizeof loading, "Loading %.256s...", G.loading);
        const char *msg = G.loading[0] ? loading : "Drop a .gltf or .glb file here, or press O to open one";
        hud_text((G.fb_w - hud_text_width(msg)) * 0.5f, G.fb_h * 0.5f - lh * 0.5f, msg, white);
    }
    if (G.error[0]) {
        hud_text(pad, y, G.error, red);
        y += lh;
    }

    if (G.show_help) {
        int lines = 1;
        for (const char *c = HELP_TEXT; *c; c++) lines += *c == '\n';
        float w = hud_text_width(HELP_TEXT) + pad * 2;
        float h = lines * lh + pad * 2 - (lh - 7 * hud_scale());
        float x = G.fb_w - w - pad, yy = pad;
        hud_rect(x, yy, w, h, panel);
        hud_text(x + pad, yy + pad, HELP_TEXT, white);
    } else {
        const char *hint = "H: help";
        hud_text(G.fb_w - hud_text_width(hint) - pad, G.fb_h - pad - 7 * hud_scale(), hint, dim);
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering

static int cmp_depth_desc(const void *a, const void *b) {
    float da = ((const DrawItem *)a)->depth, db = ((const DrawItem *)b)->depth;
    return da < db ? 1 : da > db ? -1 : 0;
}

static void collect_node(Model *m, int ni, vec3 eye, int depth) {
    if (depth > 256) return;
    Node *n = &m->nodes[ni];
    if (n->mesh >= 0) {
        MeshRef *mr = &m->meshes[n->mesh];
        for (int pi = mr->first; pi < mr->first + mr->count; pi++) {
            if (G.nitems == G.cap_items) {
                G.cap_items = G.cap_items ? G.cap_items * 2 : 256;
                G.items = realloc(G.items, G.cap_items * sizeof(DrawItem));
            }
            Prim *p = &m->prims[pi];
            DrawItem *it = &G.items[G.nitems++];
            it->prim = pi;
            it->node = ni;
            it->skinned = n->skin >= 0 && p->has_skin_attrs;
            vec3 c = v3((p->bmin[0] + p->bmax[0]) * 0.5f, (p->bmin[1] + p->bmax[1]) * 0.5f,
                        (p->bmin[2] + p->bmax[2]) * 0.5f);
            vec3 wc = it->skinned ? v3_scale(v3_add(m->bmin, m->bmax), 0.5f) : m4_point(&n->world, c);
            it->depth = v3_len(v3_sub(wc, eye));
        }
    }
    for (int c = 0; c < n->nchildren; c++) collect_node(m, n->children[c], eye, depth + 1);
}

static void draw_item(Model *m, const DrawItem *it) {
    Prim *p = &m->prims[it->prim];
    Node *n = &m->nodes[it->node];
    static const Material def = {
        {0.8f, 0.8f, 0.8f, 1}, 0.0f, 0.6f, 1, 1, {0, 0, 0}, {-1, -1, -1, -1, -1}, {0}, ALPHA_OPAQUE, 0.5f, 0, 0};
    const Material *mt = p->material >= 0 && p->material < m->nmats ? &m->mats[p->material] : &def;

    glUniform4fv(U.base_color, 1, mt->base);
    glUniform1f(U.metallic, mt->metallic);
    glUniform1f(U.roughness, mt->roughness);
    glUniform1f(U.normal_scale, mt->normal_scale);
    glUniform1f(U.occ_strength, mt->occ_strength);
    glUniform3fv(U.emissive, 1, mt->emissive);
    glUniform1i(U.alpha_mode, mt->alpha_mode);
    glUniform1f(U.alpha_cutoff, mt->cutoff);
    glUniform1i(U.unlit, mt->unlit);
    glUniform1i(U.has_tangent, p->has_tangent);
    for (int k = 0; k < TEX_COUNT; k++) {
        GLuint id = get_texture(m, mt->tex[k], k == TEX_BASE || k == TEX_EMIS);
        glActiveTexture(GL_TEXTURE0 + k);
        glBindTexture(GL_TEXTURE_2D, id ? id : G.white_tex);
        glUniform1i(U.has[k], id ? 1 : 0);
        glUniform1i(U.uvs[k], mt->uv[k]);
    }

    float det = 1;
    if (it->skinned) {
        Skin *s = &m->skins[n->skin];
        glActiveTexture(GL_TEXTURE0 + 7);
        glBindTexture(GL_TEXTURE_BUFFER, s->tex);
        glUniform1i(U.skinned, 1);
    } else {
        float nm[9];
        det = m4_normal_matrix(&n->world, nm);
        glUniformMatrix4fv(U.model, 1, GL_FALSE, n->world.m);
        glUniformMatrix3fv(U.nmat, 1, GL_FALSE, nm);
        glUniform1i(U.skinned, 0);
    }
    if (mt->double_sided) glDisable(GL_CULL_FACE);
    else glEnable(GL_CULL_FACE);
    glFrontFace(det < 0 ? GL_CW : GL_CCW);

    glBindVertexArray(p->vao);
    if (p->indexed) glDrawElements(p->mode, p->count, GL_UNSIGNED_INT, 0);
    else glDrawArrays(p->mode, 0, p->count);
}

static void set_lights(vec3 eye) {
    // Light rig is defined relative to the camera so the model is always well lit.
    vec3 fwd = v3_norm(v3_sub(G.target, eye));
    vec3 right = v3_norm(v3_cross(fwd, v3(0, 1, 0)));
    vec3 up = v3_cross(right, fwd);
    vec3 back = v3_scale(fwd, -1);
    vec3 key = v3_norm(v3_add(v3_add(v3_scale(right, -0.6f), v3_scale(up, 0.9f)), v3_scale(back, 0.7f)));
    vec3 fill = v3_norm(v3_add(v3_add(v3_scale(right, 0.9f), v3_scale(up, 0.1f)), v3_scale(back, 0.5f)));
    vec3 rim = v3_norm(v3_add(v3_scale(up, 0.6f), v3_scale(fwd, 1.0f)));
    float dirs[9] = {key.x, key.y, key.z, fill.x, fill.y, fill.z, rim.x, rim.y, rim.z};
    float cols[9] = {2.6f, 2.5f, 2.35f, 0.7f, 0.75f, 0.85f, 1.3f, 1.3f, 1.4f};
    glUniform3fv(U.light_dir, 3, dirs);
    glUniform3fv(U.light_col, 3, cols);
}

static void render(void) {
    glViewport(0, 0, G.fb_w, G.fb_h);
    glClearColor(0.07f, 0.075f, 0.09f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // background gradient
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(G.bg_prog);
    glUniform3f(glGetUniformLocation(G.bg_prog, "u_top"), 0.20f, 0.215f, 0.25f);
    glUniform3f(glGetUniformLocation(G.bg_prog, "u_bottom"), 0.06f, 0.065f, 0.08f);
    glBindVertexArray(G.empty_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    Model *m = G.model;
    if (m) {
        vec3 eye = camera_eye();
        float radius = v3_len(v3_sub(m->bmax, m->bmin)) * 0.5f;
        if (radius < 1e-6f) radius = 1;
        float aspect = G.fb_h > 0 ? (float)G.fb_w / G.fb_h : 1;
        float nearp = fmaxf(G.dist - radius * 3, G.dist * 0.01f);
        nearp = fmaxf(nearp, 1e-4f);
        float farp = G.dist + radius * 12;
        mat4 proj = m4_perspective(FOV_Y, aspect, nearp, farp);
        mat4 view = m4_lookat(eye, G.target, v3(0, 1, 0));
        mat4 vp = m4_mul(proj, view);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(G.mesh_prog);
        glUniformMatrix4fv(U.viewproj, 1, GL_FALSE, vp.m);
        glUniform3f(U.cam_pos, eye.x, eye.y, eye.z);
        glUniform3f(U.sky, 0.55f, 0.58f, 0.65f);
        glUniform3f(U.horizon, 0.42f, 0.42f, 0.44f);
        glUniform3f(U.ground, 0.16f, 0.15f, 0.14f);
        glUniform1f(U.exposure, 1.0f);
        set_lights(eye);
        glActiveTexture(GL_TEXTURE0 + 7);
        glBindTexture(GL_TEXTURE_BUFFER, G.dummy_tbo_tex);

        G.nitems = 0;
        for (int i = 0; i < m->nroots; i++) collect_node(m, m->roots[i], eye, 0);

        if (G.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // opaque + mask
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        for (int i = 0; i < G.nitems; i++) {
            Prim *p = &m->prims[G.items[i].prim];
            if (p->material >= 0 && p->material < m->nmats && m->mats[p->material].alpha_mode == ALPHA_BLEND) continue;
            draw_item(m, &G.items[i]);
        }
        // ground (shadow + grid), drawn before transparent geometry
        if (G.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1, 1);
        glUseProgram(G.grid_prog);
        vec3 gc = v3((m->bmin.x + m->bmax.x) * 0.5f, m->bmin.y, (m->bmin.z + m->bmax.z) * 0.5f);
        float cell = powf(10.0f, floorf(log10f(radius * 0.5f)));
        glUniformMatrix4fv(glGetUniformLocation(G.grid_prog, "u_viewproj"), 1, GL_FALSE, vp.m);
        glUniform3f(glGetUniformLocation(G.grid_prog, "u_center"), gc.x, gc.y, gc.z);
        glUniform1f(glGetUniformLocation(G.grid_prog, "u_extent"), radius * 5);
        glUniform1f(glGetUniformLocation(G.grid_prog, "u_cell"), cell);
        glUniform1f(glGetUniformLocation(G.grid_prog, "u_radius"), radius);
        glUniform2f(glGetUniformLocation(G.grid_prog, "u_shadow"), fmaxf((m->bmax.x - m->bmin.x) * 0.6f, radius * 0.2f),
                    fmaxf((m->bmax.z - m->bmin.z) * 0.6f, radius * 0.2f));
        glUniform1i(glGetUniformLocation(G.grid_prog, "u_show_grid"), G.show_grid);
        glUniform3f(glGetUniformLocation(G.grid_prog, "u_line_col"), 0.6f, 0.62f, 0.68f);
        if (eye.y > gc.y) {
            glBindVertexArray(G.grid_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        glDisable(GL_POLYGON_OFFSET_FILL);

        // blended geometry, back to front
        if (G.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glUseProgram(G.mesh_prog);
        qsort(G.items, G.nitems, sizeof(DrawItem), cmp_depth_desc);
        for (int i = 0; i < G.nitems; i++) {
            Prim *p = &m->prims[G.items[i].prim];
            if (!(p->material >= 0 && p->material < m->nmats && m->mats[p->material].alpha_mode == ALPHA_BLEND))
                continue;
            draw_item(m, &G.items[i]);
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDepthMask(GL_TRUE);
    }

    // HUD
    build_hud();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(G.hud_prog);
    glUniform2f(glGetUniformLocation(G.hud_prog, "u_size"), (float)G.fb_w, (float)G.fb_h);
    glBindVertexArray(G.hud_vao);
    glBindBuffer(GL_ARRAY_BUFFER, G.hud_vbo);
    glBufferData(GL_ARRAY_BUFFER, hud_n * sizeof(HudVert), hud_verts, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, hud_n);
    glBindVertexArray(0);
}

// ---------------------------------------------------------------------------------------------
// App

static void update_title(void) {
    char t[512];
    if (G.model) snprintf(t, sizeof t, "%.400s — glTF Viewer", basename_of(G.model->path));
    else snprintf(t, sizeof t, "glTF Viewer");
    SDL_SetWindowTitle(G.win, t);
}

static void open_path(const char *path) {
    if (strncmp(path, "file://", 7) == 0) {
        static char decoded[4096];
        snprintf(decoded, sizeof decoded, "%s", path + 7);
        cgltf_decode_uri(decoded);
        path = decoded;
    }
    char err[512] = {0};
    Model *m = load_model(path, err, sizeof err);
    if (!m) {
        snprintf(G.error, sizeof G.error, "%s", err);
        fprintf(stderr, "%s\n", err);
        return;
    }
    free_model(G.model);
    G.model = m;
    G.error[0] = 0;
    G.anim = 0;
    G.anim_time = 0;
    G.playing = m->nanims > 0;
    reset_view();
    update_title();
}

static void SDLCALL file_dialog_cb(void *userdata, const char *const *filelist, int filter) {
    (void)userdata;
    (void)filter;
    if (!filelist || !filelist[0]) return;
    SDL_Event e = {0};
    e.type = SDL_EVENT_USER;
    e.user.code = USER_EVENT_OPEN_FILE;
    e.user.data1 = SDL_strdup(filelist[0]);
    SDL_PushEvent(&e);
}

static void show_open_dialog(void) {
    static const SDL_DialogFileFilter filters[] = {{"glTF models", "gltf;glb"}, {"All files", "*"}};
    SDL_ShowOpenFileDialog(file_dialog_cb, NULL, G.win, filters, 2, NULL, false);
}

static void update_sizes(void) {
    SDL_GetWindowSizeInPixels(G.win, &G.fb_w, &G.fb_h);
    SDL_GetWindowSize(G.win, &G.win_w, &G.win_h);
    G.density = G.win_w > 0 ? (float)G.fb_w / G.win_w : 1.0f;
    if (G.density < 1) G.density = 1;
}

static void init_gl_objects(void) {
    G.mesh_prog = link_program(MESH_VS, MESH_FS);
    G.bg_prog = link_program(BG_VS, BG_FS);
    G.grid_prog = link_program(GRID_VS, GRID_FS);
    G.hud_prog = link_program(HUD_VS, HUD_FS);

    GLuint p = G.mesh_prog;
#define LOC(field, name) U.field = glGetUniformLocation(p, name)
    LOC(model, "u_model");
    LOC(nmat, "u_nmat");
    LOC(viewproj, "u_viewproj");
    LOC(skinned, "u_skinned");
    LOC(joint_tex, "u_joint_tex");
    LOC(base_color, "u_base_color");
    LOC(metallic, "u_metallic");
    LOC(roughness, "u_roughness");
    LOC(normal_scale, "u_normal_scale");
    LOC(occ_strength, "u_occ_strength");
    LOC(emissive, "u_emissive");
    LOC(alpha_mode, "u_alpha_mode");
    LOC(alpha_cutoff, "u_alpha_cutoff");
    LOC(unlit, "u_unlit");
    LOC(has_tangent, "u_has_tangent");
    LOC(cam_pos, "u_cam_pos");
    LOC(light_dir, "u_light_dir");
    LOC(light_col, "u_light_col");
    LOC(sky, "u_sky");
    LOC(horizon, "u_horizon");
    LOC(ground, "u_ground");
    LOC(exposure, "u_exposure");
#undef LOC
    static const char *tn[TEX_COUNT] = {"t_base", "t_mr", "t_normal", "t_occ", "t_emis"};
    static const char *hn[TEX_COUNT] = {"u_has_base", "u_has_mr", "u_has_normal", "u_has_occ", "u_has_emis"};
    static const char *un[TEX_COUNT] = {"u_uv_base", "u_uv_mr", "u_uv_normal", "u_uv_occ", "u_uv_emis"};
    glUseProgram(p);
    for (int k = 0; k < TEX_COUNT; k++) {
        U.t[k] = glGetUniformLocation(p, tn[k]);
        U.has[k] = glGetUniformLocation(p, hn[k]);
        U.uvs[k] = glGetUniformLocation(p, un[k]);
        glUniform1i(U.t[k], k);
    }
    glUniform1i(U.joint_tex, 7);

    glGenVertexArrays(1, &G.empty_vao);

    static const float quad[12] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    glGenVertexArrays(1, &G.grid_vao);
    glBindVertexArray(G.grid_vao);
    glGenBuffers(1, &G.grid_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, G.grid_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenVertexArrays(1, &G.hud_vao);
    glBindVertexArray(G.hud_vao);
    glGenBuffers(1, &G.hud_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, G.hud_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(HudVert), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(HudVert), (void *)offsetof(HudVert, rgba));
    glBindVertexArray(0);

    static const unsigned char white[4] = {255, 255, 255, 255};
    glGenTextures(1, &G.white_tex);
    glBindTexture(GL_TEXTURE_2D, G.white_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    static const float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    glGenBuffers(1, &G.dummy_tbo);
    glBindBuffer(GL_TEXTURE_BUFFER, G.dummy_tbo);
    glBufferData(GL_TEXTURE_BUFFER, sizeof ident, ident, GL_STATIC_DRAW);
    glGenTextures(1, &G.dummy_tbo_tex);
    glBindTexture(GL_TEXTURE_BUFFER, G.dummy_tbo_tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, G.dummy_tbo);

    // Defaults for attributes a primitive may not provide.
    glVertexAttrib3f(1, 0, 1, 0);
    glVertexAttrib2f(2, 0, 0);
    glVertexAttrib2f(3, 0, 0);
    glVertexAttrib4f(4, 1, 1, 1, 1);
    glVertexAttrib4f(5, 0, 0, 0, 0);
    glVertexAttrib4f(6, 0, 0, 0, 0);
    glVertexAttrib4f(7, 1, 0, 0, 1);
}

// Render one frame offscreen and write it to a PNG (used for `--screenshot out.png`).
static int write_screenshot(const char *out, int w, int h) {
    GLuint fbo, rb[2];
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(2, rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[0]);
    glBindRenderbuffer(GL_RENDERBUFFER, rb[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb[1]);
    G.fb_w = G.win_w = w;
    G.fb_h = G.win_h = h;
    G.density = 1;
    frame_model();
    if (G.model) {
        while (pump_textures(G.model, 1000)) SDL_Delay(1);
        evaluate_scene(G.model);
        update_skins(G.model);
    }
    render();
    unsigned char *px = malloc((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
    stbi_flip_vertically_on_write(1);
    int ok = stbi_write_png(out, w, h, 4, px, w * 4);
    free(px);
    if (!ok) fprintf(stderr, "could not write %s\n", out);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *file = NULL, *screenshot = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc) screenshot = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: gltfviewer [--screenshot out.png] [model.gltf|model.glb]\n");
            return 0;
        } else file = argv[i];
    }
    SDL_SetAppMetadata("glTF Viewer", "1.0", "gltfviewer");
    SDL_SetHint(SDL_HINT_APP_ID, "gltfviewer");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_WindowFlags flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (screenshot) flags |= SDL_WINDOW_HIDDEN;
    G.win = SDL_CreateWindow("glTF Viewer", 1280, 800, flags);
    if (!G.win) {
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);
        G.win = SDL_CreateWindow("glTF Viewer", 1280, 800, flags);
    }
    if (!G.win) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GLContext ctx = SDL_GL_CreateContext(G.win);
    if (!ctx) {
        fprintf(stderr, "Could not create an OpenGL 3.3 context: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_MakeCurrent(G.win, ctx);
    SDL_GL_SetSwapInterval(1);
    glEnable(GL_MULTISAMPLE);

    update_sizes();
    init_gl_objects();
    G.show_grid = 1;
    reset_view();
    if (file && !screenshot) {
        // Put the window on screen before parsing the file, so it appears instantly.
        snprintf(G.loading, sizeof G.loading, "%s", basename_of(file));
        render();
        SDL_GL_SwapWindow(G.win);
    }
    if (file) open_path(file);
    G.loading[0] = 0;
    if (screenshot) return write_screenshot(screenshot, 1280, 800);

    int running = 1, dirty = 1;
    Uint64 last = SDL_GetPerformanceCounter();
    Uint64 last_click = 0;
    while (running) {
        int loading = G.model && G.model->tex_finished < G.model->tex_needed;
        int animating = (G.model && G.playing && G.model->nanims > 0) || G.auto_rotate || loading;
        SDL_Event e;
        int have = animating || dirty ? SDL_PollEvent(&e) : SDL_WaitEventTimeout(&e, 1000);
        while (have) {
            int ev_dirty = 1;
            switch (e.type) {
            case SDL_EVENT_QUIT: running = 0; break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESIZED: update_sizes(); break;
            case SDL_EVENT_DROP_FILE:
                if (e.drop.data) open_path(e.drop.data);
                break;
            case SDL_EVENT_USER:
                if (e.user.code == USER_EVENT_OPEN_FILE && e.user.data1) {
                    open_path((const char *)e.user.data1);
                    SDL_free(e.user.data1);
                }
                break;
            case SDL_EVENT_MOUSE_MOTION: {
                SDL_MouseButtonFlags b = e.motion.state;
                int shift = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
                if ((b & SDL_BUTTON_LMASK) && !shift) {
                    G.yaw -= e.motion.xrel * 0.008f;
                    G.pitch += e.motion.yrel * 0.008f;
                    if (G.pitch > 1.55f) G.pitch = 1.55f;
                    if (G.pitch < -1.55f) G.pitch = -1.55f;
                } else if (b & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK | SDL_BUTTON_LMASK)) {
                    vec3 eye = camera_eye();
                    vec3 fwd = v3_norm(v3_sub(G.target, eye));
                    vec3 right = v3_norm(v3_cross(fwd, v3(0, 1, 0)));
                    vec3 up = v3_cross(right, fwd);
                    float k = G.dist * 2 * tanf(FOV_Y * 0.5f) / (G.win_h > 0 ? G.win_h : 1);
                    G.target = v3_add(G.target, v3_add(v3_scale(right, -e.motion.xrel * k), v3_scale(up, e.motion.yrel * k)));
                } else {
                    ev_dirty = 0;
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL: {
                float f = powf(0.88f, e.wheel.y);
                G.dist *= f;
                float lo = G.frame_dist * 0.001f, hi = G.frame_dist * 50;
                if (G.dist < lo) G.dist = lo;
                if (G.dist > hi) G.dist = hi;
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (e.button.button == SDL_BUTTON_LEFT) {
                    Uint64 now = SDL_GetTicks();
                    if (e.button.clicks >= 2 || now - last_click < 300) frame_model();
                    last_click = now;
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                switch (e.key.key) {
                case SDLK_ESCAPE:
                    if (G.fullscreen) {
                        G.fullscreen = 0;
                        SDL_SetWindowFullscreen(G.win, false);
                    } else {
                        running = 0;
                    }
                    break;
                case SDLK_Q: running = 0; break;
                case SDLK_F: frame_model(); break;
                case SDLK_R: reset_view(); break;
                case SDLK_A: G.auto_rotate = !G.auto_rotate; break;
                case SDLK_G: G.show_grid = !G.show_grid; break;
                case SDLK_W: G.wireframe = !G.wireframe; break;
                case SDLK_H: G.show_help = !G.show_help; break;
                case SDLK_O: show_open_dialog(); break;
                case SDLK_F11:
                    G.fullscreen = !G.fullscreen;
                    SDL_SetWindowFullscreen(G.win, G.fullscreen);
                    break;
                case SDLK_SPACE: G.playing = !G.playing; break;
                case SDLK_N:
                case SDLK_P:
                    if (G.model && G.model->nanims > 0) {
                        int d = e.key.key == SDLK_N ? 1 : -1;
                        G.anim = (G.anim + d + G.model->nanims) % G.model->nanims;
                        G.anim_time = 0;
                        G.playing = 1;
                        reset_pose(G.model);
                    }
                    break;
                default: break;
                }
                break;
            default: break;
            }
            dirty |= ev_dirty;
            have = SDL_PollEvent(&e);
        }
        if (!running) break;

        Uint64 now = SDL_GetPerformanceCounter();
        double dt = (double)(now - last) / (double)SDL_GetPerformanceFrequency();
        last = now;
        if (dt > 0.1) dt = 0.1;

        if (G.auto_rotate) G.yaw += (float)dt * 0.5f;
        Model *m = G.model;
        if (m) {
            int before = m->tex_finished;
            pump_textures(m, 8);
            if (m->tex_finished != before) dirty = 1;
            if (m->nanims > 0) {
                Anim *a = &m->anims[G.anim];
                if (G.playing) G.anim_time += dt;
                if (a->duration > 0) G.anim_time = fmod(G.anim_time, a->duration);
                apply_anim(m, G.anim, (float)G.anim_time);
            }
            evaluate_scene(m);
            update_skins(m);
        }
        if (dirty || animating) {
            render();
            SDL_GL_SwapWindow(G.win);
            dirty = 0;
        }
    }

    free_model(G.model);
    SDL_GL_DestroyContext(ctx);
    SDL_DestroyWindow(G.win);
    SDL_Quit();
    return 0;
}
