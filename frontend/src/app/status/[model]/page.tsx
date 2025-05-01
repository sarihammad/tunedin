'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { fetchFromAPI } from '@/lib/api';

interface ModelStatus {
  model_name: string;
  is_trained: boolean;
  last_trained: string | null;
  loss_history?: number[];
}

export default function ModelStatusPage() {
  const { model } = useParams();
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetchFromAPI(`/status/${model}`);

        if (!res.ok) {
          throw new Error(`Error: ${res.status}`);
        }

        const data = await res.json();
        setStatus(data);
      } catch (err: any) {
        setError(err.message);
      }
    }

    fetchStatus();
  }, [model]);

  if (error) return <div className="text-red-500">Error: {error}</div>;
  if (!status) return <div>Loading model status...</div>;

  return (
    <div className="max-w-3xl mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-4">Model: {status.model_name}</h1>
      <p className="mb-2">
        <strong>Status:</strong> {status.is_trained ? 'Trained' : 'Not Trained'}
      </p>
      <p className="mb-4">
        <strong>Last Trained:</strong>{' '}
        {status.last_trained ? new Date(status.last_trained).toLocaleString() : 'Never'}
      </p>

      {status.loss_history && status.loss_history.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Loss History</h2>
          <ul className="list-disc pl-5 space-y-1 text-sm text-gray-600">
            {status.loss_history.slice(-10).map((loss, index) => (
              <li key={index}>Epoch {status.loss_history!.length - 10 + index + 1}: {loss.toFixed(4)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
