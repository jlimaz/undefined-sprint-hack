import { withAui } from "@assistant-ui/next";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The default corner sits on top of the Library's Run button.
  devIndicators: { position: "bottom-right" },
};

export default withAui(nextConfig);
