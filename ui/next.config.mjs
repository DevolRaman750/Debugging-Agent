/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow requests to the Python backend
  async rewrites() {
    return [
      // Optional: Direct proxy for Python backend (if you want /backend/* to forward)
      // {
      //   source: "/backend/:path*",
      //   destination: `${process.env.REST_API_ENDPOINT || "http://localhost:8000"}/:path*`,
      // },
    ];
  },

  // Disable strict mode double-renders in dev (optional)
  reactStrictMode: true,

  // Allow images from external sources if needed
  images: {
    remotePatterns: [],
  },
};

export default nextConfig;
