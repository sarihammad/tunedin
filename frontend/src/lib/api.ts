export async function fetchFromAPI(path: string, options: RequestInit = {}) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  const apiKey = process.env.NEXT_PUBLIC_API_KEY

  const response = await fetch(`${apiUrl}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-KEY': apiKey ?? '',
      ...(options.headers || {}),
    },
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'API Error')
  }

  return response.json()
}