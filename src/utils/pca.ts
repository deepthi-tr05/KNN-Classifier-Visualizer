// Simple PCA implementation in TypeScript

function mean(data: number[][]): number[] {
  const n = data.length;
  const m = data[0].length;
  const means = new Array(m).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      means[j] += data[i][j];
    }
  }
  return means.map((v) => v / n);
}

function centerData(data: number[][], means: number[]): number[][] {
  return data.map((row) => row.map((v, j) => v - means[j]));
}

function covarianceMatrix(centered: number[][]): number[][] {
  const n = centered.length;
  const m = centered[0].length;
  const cov: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centered[k][i] * centered[k][j];
      }
      cov[i][j] = sum / (n - 1);
      cov[j][i] = cov[i][j];
    }
  }
  return cov;
}

// Jacobi eigenvalue algorithm for symmetric matrices
function jacobiEigen(
  matrix: number[][],
  maxIter = 100
): { eigenvalues: number[]; eigenvectors: number[][] } {
  const n = matrix.length;
  const A = matrix.map((row) => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxVal) {
          maxVal = Math.abs(A[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-10) break;

    // Compute rotation
    const theta =
      Math.abs(A[p][p] - A[q][q]) < 1e-10
        ? Math.PI / 4
        : 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Apply rotation to A
    const App = c * c * A[p][p] + 2 * s * c * A[p][q] + s * s * A[q][q];
    const Aqq = s * s * A[p][p] - 2 * s * c * A[p][q] + c * c * A[q][q];

    A[p][q] = 0;
    A[q][p] = 0;
    A[p][p] = App;
    A[q][q] = Aqq;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const Aip = c * A[i][p] + s * A[i][q];
        const Aiq = -s * A[i][p] + c * A[i][q];
        A[i][p] = Aip;
        A[p][i] = Aip;
        A[i][q] = Aiq;
        A[q][i] = Aiq;
      }
    }

    // Update eigenvectors
    for (let i = 0; i < n; i++) {
      const Vip = c * V[i][p] + s * V[i][q];
      const Viq = -s * V[i][p] + c * V[i][q];
      V[i][p] = Vip;
      V[i][q] = Viq;
    }
  }

  const eigenvalues = Array.from({ length: n }, (_, i) => A[i][i]);
  return { eigenvalues, eigenvectors: V };
}

export interface PCAResult {
  transformed: number[][];
  explainedVariance: number[];
  explainedVarianceRatio: number[];
  components: number[][];
  means: number[];
}

export function computePCA(data: number[][], nComponents = 2): PCAResult {
  const means = mean(data);
  const centered = centerData(data, means);
  const cov = covarianceMatrix(centered);
  const { eigenvalues, eigenvectors } = jacobiEigen(cov);

  // Sort by eigenvalue descending
  const indices = eigenvalues
    .map((val, idx) => ({ val, idx }))
    .sort((a, b) => b.val - a.val)
    .map((x) => x.idx);

  const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
  const sortedEigenvalues = indices.map((i) => eigenvalues[i]);
  const explainedVarianceRatio = sortedEigenvalues.map((v) => v / totalVariance);

  // Extract top components
  const components: number[][] = [];
  for (let c = 0; c < nComponents; c++) {
    const col = indices[c];
    components.push(eigenvectors.map((row) => row[col]));
  }

  // Project data
  const transformed = centered.map((row) => {
    const projected: number[] = [];
    for (let c = 0; c < nComponents; c++) {
      let sum = 0;
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * components[c][j];
      }
      projected.push(sum);
    }
    return projected;
  });

  return {
    transformed,
    explainedVariance: sortedEigenvalues.slice(0, nComponents),
    explainedVarianceRatio: explainedVarianceRatio.slice(0, nComponents),
    components: components.slice(0, nComponents),
    means,
  };
}
