#include "halfedge.h"
#include <iostream>

void HalfEdgeRepr::edgeFlip(HalfEdge *he)
{
    Face *f_l = he->face;
    Face *f_r = he->twin->face;

    HalfEdge *he_tb = he;
    HalfEdge *he_lt = he->next;
    HalfEdge *he_tr = he->next->next;

    HalfEdge *he_bt = he->twin;
    HalfEdge *he_rb = he->twin->next;
    HalfEdge *he_bl = he->twin->next->next;

    Vertex *v_t = he_tr->vertex;
    Vertex *v_b = he_bl->vertex;
    Vertex *v_l = he_lt->vertex;
    Vertex *v_r = he_rb->vertex;

    // Reassign vertices
    he_tb->vertex = v_t;
    he_bt->vertex = v_b;
    v_l->halfEdge = he_lt;
    v_r->halfEdge = he_rb;

    // Reassign next pointers
    he_tb->next = he_bl;
    he_bl->next = he_lt;
    he_lt->next = he_tb;

    he_bt->next = he_tr;
    he_tr->next = he_rb;
    he_rb->next = he_bt;

    // Reassign faces
    he_tr->face = f_r;
    he_bl->face = f_l;

    f_r->halfEdge = he_tr;
    f_l->halfEdge = he_bl;
}

void HalfEdgeRepr::edgeSplit(HalfEdge *he)
{
    HalfEdge *he_cl = he;
    HalfEdge *he_lt = he->next;
    HalfEdge *he_tr = he->next->next;
    HalfEdge *he_cr = he->twin;
    HalfEdge *he_rb = he->twin->next;
    HalfEdge *he_bl = he->twin->next->next;
    auto he_lc = std::make_unique<HalfEdge>(_half_edges.size());
    auto he_cb = std::make_unique<HalfEdge>(_half_edges.size() + 1);
    auto he_bc = std::make_unique<HalfEdge>(_half_edges.size() + 2);
    auto he_ct = std::make_unique<HalfEdge>(_half_edges.size() + 3);
    auto he_tc = std::make_unique<HalfEdge>(_half_edges.size() + 4);
    auto he_rc = std::make_unique<HalfEdge>(_half_edges.size() + 5);

    Face *f_tl = he->face;
    Face *f_br = he->twin->face;
    auto f_tr = std::make_unique<Face>(_faces.size());
    auto f_bl = std::make_unique<Face>(_faces.size() + 1);

    Vertex *vt = he_tr->vertex;
    Vertex *vl = he_cr->vertex;
    Vertex *vr = he_cl->vertex;
    Vertex *vb = he_bl->vertex;

    auto vnew = std::make_unique<Vertex>(_vertices.size(), (vl->position + vr->position) / 2.0f);

    Edge *edge_l = he_cl->edge;
    auto edge_r = std::make_unique<Edge>(_edges.size(), he_cr);
    auto edge_t = std::make_unique<Edge>(_edges.size() + 1, he_tc.get(), true);
    auto edge_b = std::make_unique<Edge>(_edges.size() + 2, he_bc.get(), true);

    // Resassign twins
    he_bc->twin = he_cb.get();
    he_cb->twin = he_bc.get();
    he_rc->twin = he_cr;
    he_cr->twin = he_rc.get();
    he_cl->twin = he_lc.get();
    he_lc->twin = he_cl;
    he_ct->twin = he_tc.get();
    he_tc->twin = he_ct.get();

    // Reassign next pointers
    he_cl->next = he_lt;
    he_lt->next = he_tc.get();
    he_tc->next = he_cl;
    he_tr->next = he_rc.get();
    he_rc->next = he_ct.get();
    he_ct->next = he_tr;
    he_cr->next = he_rb;
    he_rb->next = he_bc.get();
    he_bc->next = he_cr;
    he_bl->next = he_lc.get();
    he_lc->next = he_cb.get();
    he_cb->next = he_bl;

    // Reassign faces
    he_cl->face = f_tl;
    he_lt->face = f_tl;
    he_tr->face = f_tr.get();
    he_cr->face = f_br;
    he_rb->face = f_br;
    he_bl->face = f_bl.get();
    he_lc->face = f_bl.get();
    he_cb->face = f_bl.get();
    he_bc->face = f_br;
    he_ct->face = f_tr.get();
    he_tc->face = f_tl;
    he_rc->face = f_tr.get();

    // Reassign half-edge pointers for faces
    f_tl->halfEdge = he_lt;
    f_br->halfEdge = he_rb;
    f_tr->halfEdge = he_tr;
    f_bl->halfEdge = he_bl;

    // Reassign vertices
    he_cl->vertex = vnew.get();
    he_lt->vertex = vl;
    he_tr->vertex = vt;
    he_cr->vertex = vnew.get();
    he_rb->vertex = vr;
    he_bl->vertex = vb;
    he_lc->vertex = vl;
    he_cb->vertex = vnew.get();
    he_bc->vertex = vb;
    he_ct->vertex = vnew.get();
    he_tc->vertex = vt;
    he_rc->vertex = vr;

    // Reassign half-edge pointers for vertices
    vl->halfEdge = he_lt;
    vt->halfEdge = he_tr;
    vr->halfEdge = he_rb;
    vb->halfEdge = he_bl;
    vnew->halfEdge = he_cl;

    // Reassign edges
    he_lc->edge = edge_l;
    he_cl->edge = edge_l;
    he_bc->edge = edge_b.get();
    he_cb->edge = edge_b.get();
    he_tc->edge = edge_t.get();
    he_ct->edge = edge_t.get();
    he_rc->edge = edge_r.get();
    he_cr->edge = edge_r.get();

    // Reassign half-edge pointers for edges
    edge_l->halfEdge = he_cl;

    _half_edges.push_back(std::move(he_lc));
    _half_edges.push_back(std::move(he_cb));
    _half_edges.push_back(std::move(he_bc));
    _half_edges.push_back(std::move(he_ct));
    _half_edges.push_back(std::move(he_tc));
    _half_edges.push_back(std::move(he_rc));
    _edges.push_back(std::move(edge_t));
    _edges.push_back(std::move(edge_r));
    _edges.push_back(std::move(edge_b));
    _faces.push_back(std::move(f_tr));
    _faces.push_back(std::move(f_bl));
    _vertices.push_back(std::move(vnew));
}

void HalfEdgeRepr::edgeCollapse(HalfEdge *he)
{
    HalfEdge *he_lr = he;
    HalfEdge *he_rt = he_lr->next;
    HalfEdge *he_tl = he_rt->next;

    HalfEdge *he_rl = he->twin;
    HalfEdge *he_lb = he_rl->next;
    HalfEdge *he_br = he_lb->next;

    HalfEdge *he_rb = he_br->twin;
    HalfEdge *he_tr = he_rt->twin;

    HalfEdge *he_rb_next = he_rb->next;
    HalfEdge *he_rb_next_next = he_rb->next->next;
    HalfEdge *he_tr_next = he_tr->next;
    HalfEdge *he_tr_next_next = he_tr->next->next;

    // std::cout << "he_lr: " << he_lr << ", he_rt: " << he_rt << ", he_tl: " << he_tl << std::endl;
    // std::cout << "he_rl: " << he_rl << ", he_lb: " << he_lb << ", he_br: " << he_br << std::endl;
    // std::cout << "he_rb: " << he_rb << ", he_tr: " << he_tr << std::endl;

    Vertex *del_v = he_rl->vertex;
    Vertex *keep_v = he_lr->vertex;
    Vertex *v_t = he_tl->vertex;
    Vertex *v_b = he_br->vertex;

    Face *del_f_t = he_lr->face;
    Face *del_f_b = he_rl->face;
    Face *keep_f_t = he_tr->face;
    Face *keep_f_b = he_rb->face;

    HalfEdge *looper = del_v->halfEdge;
    do
    {
        looper->vertex = keep_v;
        if (!looper->twin) break;  // boundary
        looper = looper->twin->next;
    } while (looper != del_v->halfEdge);

    he_rb_next_next->next = he_lb;
    he_lb->next = he_rb_next;

    he_tr_next_next->next = he_tl;
    he_tl->next = he_tr_next == he_rb ? he_lb : he_tr_next;

    v_t->halfEdge = he_tl;
    v_b->halfEdge = he_rb_next;
    keep_v->halfEdge = he_lb;
    keep_v->position = (keep_v->position + del_v->position) / 2.0f;

    he_tl->face = keep_f_t;
    he_lb->face = keep_f_b;

    keep_f_t->halfEdge = he_tl;
    keep_f_b->halfEdge = he_lb;

    he_lr->deleted = true;
    he_rl->deleted = true;
    he_lr->edge->deleted = true;

    he_rt->deleted = true;
    he_tr->deleted = true;
    he_rt->edge->deleted = true;

    he_rb->deleted = true;
    he_br->deleted = true;
    he_rb->edge->deleted = true;

    del_v->deleted = true;
    del_f_t->deleted = true;
    del_f_b->deleted = true;
}
