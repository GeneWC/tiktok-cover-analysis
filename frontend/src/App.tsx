import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import UploadPage from "./pages/UploadPage";
import ProcessingPage from "./pages/ProcessingPage";
import ReportPage from "./pages/ReportPage";
import ChannelUploadPage from "./pages/ChannelUploadPage";
import ChannelProcessingPage from "./pages/ChannelProcessingPage";
import ChannelReportPage from "./pages/ChannelReportPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/processing/:id" element={<ProcessingPage />} />
        <Route path="/report/:id" element={<ReportPage />} />
        <Route path="/channel" element={<ChannelUploadPage />} />
        <Route
          path="/channel/processing/:id"
          element={<ChannelProcessingPage />}
        />
        <Route path="/channel/report/:id" element={<ChannelReportPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
