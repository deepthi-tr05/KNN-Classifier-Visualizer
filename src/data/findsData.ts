export const ATTRIBUTES = [
  'Sky',
  'AirTemp',
  'Humidity',
  'Wind',
  'Water',
  'Forecast',
];

export type Example = {
  sky: string;
  airTemp: string;
  humidity: string;
  wind: string;
  water: string;
  forecast: string;
  target: 'Yes' | 'No';
};

// Standard "Enjoy Sport" dataset
export const findsDataset: Example[] = [
  { sky: 'Sunny', airTemp: 'Warm', humidity: 'Normal', wind: 'Strong', water: 'Warm', forecast: 'Same', target: 'Yes' },
  { sky: 'Sunny', airTemp: 'Warm', humidity: 'High', wind: 'Strong', water: 'Warm', forecast: 'Same', target: 'Yes' },
  { sky: 'Rainy', airTemp: 'Cold', humidity: 'High', wind: 'Strong', water: 'Warm', forecast: 'Change', target: 'No' },
  { sky: 'Sunny', airTemp: 'Warm', humidity: 'High', wind: 'Strong', water: 'Cool', forecast: 'Change', target: 'Yes' },
];

export function exampleToArray(e: Example): string[] {
  return [e.sky, e.airTemp, e.humidity, e.wind, e.water, e.forecast];
}

export type HypothesisState = {
  step: number; // example index
  hypothesis: string[];
  event: 'init' | 'positive-match' | 'positive-mismatch' | 'skipped';
  changedAttrs: boolean[];
  exampleValues: string[];
};

export function computeFindSSteps(): HypothesisState[] {
  const steps: HypothesisState[] = [];
  let hypothesis: string[] | null = null;

  findsDataset.forEach((ex, i) => {
    const vals = exampleToArray(ex);

    if (ex.target === 'Yes') {
      if (hypothesis === null) {
        hypothesis = [...vals];
        steps.push({
          step: i,
          hypothesis: [...hypothesis],
          event: 'init',
          changedAttrs: hypothesis.map(() => true),
          exampleValues: vals,
        });
      } else {
        const changed = hypothesis.map((h, idx) => h !== vals[idx]);
        hypothesis = hypothesis.map((h, idx) => (h === vals[idx] ? h : '?'));
        steps.push({
          step: i,
          hypothesis: [...hypothesis],
          event: 'positive-mismatch',
          changedAttrs: changed,
          exampleValues: vals,
        });
      }
    } else {
      steps.push({
        step: i,
        hypothesis: hypothesis ? [...hypothesis] : ATTRIBUTES.map(() => '∅'),
        event: 'skipped',
        changedAttrs: ATTRIBUTES.map(() => false),
        exampleValues: vals,
      });
    }
  });

  return steps;
}
