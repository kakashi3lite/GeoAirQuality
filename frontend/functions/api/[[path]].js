// Cloudflare Pages Function — reverse proxy for the GeoAirQuality API.
//
// The React SPA calls the API at same-origin `/api/v1/...` (axios baseURL
// '/api'). This function forwards those requests to the real backend so the
// browser never needs CORS and the backend host stays hidden from clients.
//
// Configure the backend origin via the `API_ORIGIN` Pages variable/secret
// (defaults to https://api.geoairquality.com). Works with GET/POST/DELETE,
// forwards query strings and request bodies, and returns a clean 502 JSON
// error if the backend is unreachable (so the SPA can show a friendly
// "data unavailable" state instead of a browser network error).

const DEFAULT_ORIGIN = 'https://api.geoairquality.com';

export async function onRequest(context) {
  const { request, env } = context;
  const origin = (env && env.API_ORIGIN) || DEFAULT_ORIGIN;

  const url = new URL(request.url);
  const target = `${origin}${url.pathname}${url.search}`;

  // Strip hop-by-hop headers that must not be forwarded upstream.
  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('connection');

  // Read the body eagerly so forwarding works identically in the Workers
  // runtime and Node-based tests (Request.body streams behave differently).
  // Our payloads are small JSON bodies (symptom logs), so this is fine.
  const body = ['GET', 'HEAD'].includes(request.method)
    ? undefined
    : await request.arrayBuffer();

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      redirect: 'manual',
    });

    // Clone upstream body so we can also strip hop-by-hop response headers.
    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.delete('transfer-encoding');
    responseHeaders.delete('connection');

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (err) {
    return Response.json(
      { detail: 'Upstream API unavailable', origin },
      { status: 502 },
    );
  }
}
