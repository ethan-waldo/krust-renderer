use crate::aabb::Aabb;
use crate::hit::HitRecord;
use crate::hit::{BoundingBox, Hittable, Object};
use crate::ray::Ray;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

#[derive(Clone)]
#[allow(dead_code)]
pub struct Bvh {
    pub objects: Vec<Arc<Object>>,
    pub time0: f64,
    pub time1: f64,
    pub left: Arc<Object>,
    pub right: Arc<Object>,
    pub bbox: Aabb,
}

impl Bvh {
    pub fn new(objects: &mut [Arc<Object>], time0: f64, time1: f64) -> Bvh {
        let span = objects.len();
        let mid = span / 2;

        if span <= 0 {
            panic!("Empty BVH");
        }

        let (left, right, bbox) = if span == 1 {
            let left = objects[0].clone();
            let right = objects[0].clone();
            let bbox = left.bounding_box(0.0, 1.0);
            (left, right, bbox)
        } else if span == 2 {
            let l = objects[0].clone();
            let r = objects[1].clone();
            let bbox = Aabb::surrounding_box(l.bounding_box(0.0, 1.0), r.bounding_box(0.0, 1.0));
            let comparator = Bvh::get_comparator(&bbox);
            if comparator(&l, &r) {
                (l, r, bbox)
            } else {
                (r, l, bbox)
            }
        } else {
            let mut bbox = objects[0].bounding_box(0.0, 1.0);
            for i in 1..span {
                bbox = Aabb::surrounding_box(bbox, objects[i].bounding_box(0.0, 1.0));
            }
            let comparator = Bvh::get_comparator(&bbox);

            objects.par_sort_by(|a, b| {
                if comparator(a, b) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });
            let (left_slice, right_slice) = objects.split_at_mut(mid);
            let left = Arc::new(Object::Bvh(Bvh::new(left_slice, time0, time1)));
            let right = Arc::new(Object::Bvh(Bvh::new(right_slice, time0, time1)));
            let lbox = left.bounding_box(time0, time1);
            let rbox = right.bounding_box(time0, time1);
            let bbox = Aabb::surrounding_box(lbox, rbox);
            (left, right, bbox)
        };

        Bvh {
            objects: objects.to_vec(),
            time0,
            time1,
            left,
            right,
            bbox,
        }
    }

    pub fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> (bool, Option<HitRecord>) {
        let bbox_hit = self.bbox.hit(r, t_min, t_max);
        match bbox_hit {
            (false, None) => return (false, None),
            _ => (),
        }
        let hit_left = self.left.hit(r, t_min, t_max);
        match hit_left {
            (true, Some(left_hit_rec)) => {
                let hit_right = self.right.hit(r, t_min, left_hit_rec.t);
                match hit_right {
                    (true, Some(right_hit_rec)) => return (true, Some(right_hit_rec)),
                    _ => return (true, Some(left_hit_rec)),
                }
            }
            _ => {
                let hit_right = self.right.hit(r, t_min, t_max);
                match hit_right {
                    (true, Some(right_hit_rec)) => return (true, Some(right_hit_rec)),
                    _ => return (false, None),
                }
            }
        }
    }

    pub fn bounding_box(&self, _time0: f64, _time1: f64) -> Aabb {
        self.bbox
    }

    pub fn box_compare(a: &Arc<Object>, b: &Arc<Object>, axis: usize) -> bool {
        let box_a = a.bounding_box(0.0, 0.0);
        let box_b = b.bounding_box(0.0, 0.0);
        if axis == 0 {
            return box_a.min.x < box_b.min.x;
        } else if axis == 1 {
            return box_a.min.y < box_b.min.y;
        } else {
            return box_a.min.z < box_b.min.z;
        }
    }

    pub fn box_x_compare(a: &Arc<Object>, b: &Arc<Object>) -> bool {
        return Bvh::box_compare(&a, &b, 0);
    }

    pub fn box_y_compare(a: &Arc<Object>, b: &Arc<Object>) -> bool {
        return Bvh::box_compare(&a, &b, 1);
    }

    pub fn box_z_compare(a: &Arc<Object>, b: &Arc<Object>) -> bool {
        return Bvh::box_compare(&a, &b, 2);
    }

    pub fn get_comparator(bbox: &Aabb) -> fn(&Arc<Object>, &Arc<Object>) -> bool {
        let axis = bbox.longest_axis();
        if axis == 0 {
            return Bvh::box_x_compare;
        } else if axis == 1 {
            return Bvh::box_y_compare;
        } else {
            return Bvh::box_z_compare;
        }
    }
}
