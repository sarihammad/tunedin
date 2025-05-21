

'use client'

import { useState } from 'react'

export default function TrainPage() {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const startTraining = async () => {
    setLoading(true)
    setMessage('')
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
      })

      if (!response.ok) throw new Error('Training failed')

      const data = await response.json()
      setMessage(data.message || 'Training completed successfully!')
    } catch (err: unknown) {
      if (err instanceof Error) {
        setMessage(err.message)
      } else {
        setMessage('An unexpected error occurred.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Train Model</h1>
      <button
        onClick={startTraining}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Training...' : 'Start Training'}
      </button>
      {message && <p className="mt-4 text-sm text-gray-700">{message}</p>}
    </div>
  )
}