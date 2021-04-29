use std::{fmt::Debug, ops::Mul};

pub struct Matrix<const R: usize, const C: usize> {
    inner: [[f64; C]; R],
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub fn new(inner: [[f64; C]; R]) -> Self {
        Self { inner }
    }

    pub fn new_zeroed() -> Self {
        let inner = [[0.0; C]; R];
        Self { inner }
    }

    pub fn set_at_position(&mut self, row: usize, col: usize, val: f64) {
        self.inner[row][col] = val;
    }

    pub fn get_at_position(&self, row: usize, col: usize) -> f64 {
        self.inner[row][col]
    }

    pub fn apply_to_each<F>(&mut self, callback: &F)
    where
        F: Fn(&f64) -> f64,
    {
        for row in self.inner.iter_mut() {
            // for each element
            for col_item in row.iter_mut() {
                *col_item = callback(&col_item);
            }
        }
    }
}

impl<const R: usize, const C: usize> Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self
            .inner
            .iter()
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

        // create resulting matrix
        let mut out = Matrix::new_zeroed();

        // fill output matrix
        for index_row in 0..RI {
            for index_col in 0..CO {
                let mut acc = 0.0;
                for i in 0..CI {
                    let current_left = self.get_at_position(index_row, i);
                    let current_right = rhs.get_at_position(i, index_col);
                    acc += current_left * current_right;
                }
                out.set_at_position(index_row, index_col, acc);
            }
            // index_col --> die momentane spalte der rechten matrix
            // index_row --> die momentane zeile der linken matrix
            // zusammen --> koordinaten für die ergebnis-matrix
        }
        out
    }
}
