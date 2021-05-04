use rayon::prelude::*;
use std::array::IntoIter;
use std::{fmt::Debug, ops::Mul};

pub struct Matrix<const R: usize, const C: usize> {
    inner: Vec<f64>,
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub unsafe fn new_unchecked(inner: Vec<Vec<f64>>) -> Self {
        let inner = inner.into_iter().map(|i| i.into_iter()).flatten().collect();

        Self { inner }
    }

    pub fn new(inner: [[f64; C]; R]) -> Self {
        let inner = IntoIter::new(inner).map(IntoIter::new).flatten().collect();

        Self { inner }
    }

    fn new_1d(inner: Vec<f64>) -> Self {
        Self { inner }
    }

    pub fn new_zeroed() -> Self {
        let inner = vec![0.0; C * R];

        Self { inner }
    }

    #[inline]
    fn calculate_index(&self, row: usize, col: usize) -> usize {
        let index = (row * C) + col;
        index
    }

    fn calculate_indices() -> Vec<(usize, usize)> {
        (0..R)
            .map(|index_row| (0..C).map(move |index_col| (index_row, index_col)))
            .flatten()
            .collect::<Vec<_>>()
    }

    pub fn set_at_position(&mut self, row: usize, col: usize, val: f64) {
        let index = self.calculate_index(row, col);
        self.inner[index] = val;
    }

    pub fn get_at_position(&self, row: usize, col: usize) -> f64 {
        self.inner[self.calculate_index(row, col)]
    }

    pub fn apply_to_each<F>(&mut self, callback: &F)
    where
        F: Fn(&f64) -> f64,
    {
        for item in self.inner.iter_mut() {
            *item = callback(&item);
        }
    }
}

impl<const R: usize, const C: usize> Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self
            .inner
            .chunks(C)
            .map(|i| {
                i.iter()
                    .map(|i1| format!("{:06.3}", i1))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n");

        write!(f, "{}", inner)
    }
}

impl<const RI: usize, const CI: usize, const RO: usize, const CO: usize> Mul<Matrix<RO, CO>>
    for Matrix<RI, CI>
{
    type Output = Matrix<RI, CO>;

    fn mul(self, rhs: Matrix<RO, CO>) -> Self::Output {
        // make sure that the dimensions are valid
        assert!(CI == RO, "Invalid dimensions!");

        let indices = Matrix::<RI, CO>::calculate_indices();

        let res = indices
            .par_iter()
            .map(|(index_row, index_col)| {
                let mut acc = 0.0;
                for i in 0..CI {
                    let current_left = self.get_at_position(*index_row, i);
                    let current_right = rhs.get_at_position(i, *index_col);
                    acc += current_left * current_right;
                }

                // index_col --> die momentane spalte der rechten matrix
                // index_row --> die momentane zeile der linken matrix
                // zusammen --> koordinaten für die ergebnis-matrix

                acc
            })
            .collect();

        Matrix::new_1d(res)
    }
}
