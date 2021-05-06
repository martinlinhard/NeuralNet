use rayon::prelude::*;
use std::{array::IntoIter, ops::Sub};
use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

pub struct Matrix {
    inner: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl Matrix {
    pub unsafe fn new_unchecked(inner: Vec<Vec<f64>>, rows: usize, columns: usize) -> Self {
        let inner = inner.into_iter().map(|i| i.into_iter()).flatten().collect();

        Self {
            inner,
            rows,
            columns,
        }
    }

    pub fn new<const R: usize, const C: usize>(inner: [[f64; C]; R]) -> Self {
        let inner = IntoIter::new(inner).map(IntoIter::new).flatten().collect();

        Self {
            inner,
            rows: R,
            columns: C,
        }
    }

    fn new_1d(inner: Vec<f64>, rows: usize, columns: usize) -> Self {
        Self {
            inner,
            rows,
            columns,
        }
    }

    pub fn new_zeroed(rows: usize, columns: usize) -> Self {
        let inner = vec![0.0; rows * columns];

        Self {
            inner,
            rows,
            columns,
        }
    }

    #[inline]
    fn calculate_index(&self, row: usize, col: usize) -> usize {
        let index = (row * self.columns) + col;
        index
    }

    fn calculate_indices(
        rows: usize,
        columns: usize,
    ) -> impl ParallelIterator<Item = (usize, usize)> {
        (0..rows)
            .into_par_iter()
            .map(move |index_row| {
                (0..columns)
                    .into_par_iter()
                    .map(move |index_col| (index_row, index_col))
            })
            .flatten()
    }

    fn calculate_indices_single(
        rows: usize,
        columns: usize,
    ) -> impl Iterator<Item = (usize, usize)> {
        (0..rows)
            .into_iter()
            .map(move |index_row| {
                (0..columns)
                    .into_iter()
                    .map(move |index_col| (index_row, index_col))
            })
            .flatten()
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
        F: Fn(&f64) -> f64 + Sync,
    {
        self.inner.par_iter_mut().for_each(|item| {
            *item = callback(&item);
        });
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self
            .inner
            .chunks(self.columns)
            .map(|i| {
                i.iter()
                    .map(|i1| format!("{:06.3}", i1))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n");

        write!(f, "Matrix ({}X{})\n{}", self.rows, self.columns, inner)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        // make sure that the dimensions are valid
        assert!(self.columns == rhs.rows, "Invalid dimensions!");

        let indices = Matrix::calculate_indices(self.rows, rhs.columns);

        let res = indices
            .map(|(index_row, index_col)| {
                let mut acc = 0.0;
                for i in 0..self.columns {
                    let current_left = self.get_at_position(index_row, i);
                    let current_right = rhs.get_at_position(i, index_col);
                    acc += current_left * current_right;
                }

                // index_col --> die momentane spalte der rechten matrix
                // index_row --> die momentane zeile der linken matrix
                // zusammen --> koordinaten für die ergebnis-matrix

                acc
            })
            .collect();

        Matrix::new_1d(res, self.rows, rhs.columns)
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        assert!(
            self.columns == rhs.columns && self.rows == rhs.rows,
            "Invalid dimensions!"
        );

        let result = self
            .inner
            .par_iter()
            .zip(rhs.inner.par_iter())
            .map(|(left, right)| left + right)
            .collect();

        Matrix::new_1d(result, self.rows, self.columns)
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        assert!(
            self.columns == rhs.columns && self.rows == rhs.rows,
            "Invalid dimensions!"
        );

        let result = self
            .inner
            .par_iter()
            .zip(rhs.inner.par_iter())
            .map(|(left, right)| left - right)
            .collect();

        Matrix::new_1d(result, self.rows, self.columns)
    }
}
