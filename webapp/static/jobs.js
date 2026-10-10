// One poll at a time. A page reload can reconnect using the saved job id.
window.HandwriterJobs = {
  async start(url, body) {
    const response = await fetch(url, {method: 'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start the job. Try again.');
    return data.job_id;
  },
  async wait(id, onProgress) {
    let failures = 0;
    while (true) {
      let response;
      try { response = await fetch(`/jobs/${encodeURIComponent(id)}`); }
      catch (error) {
        if (++failures >= 5) throw new Error('Connection lost. Reload this page to reconnect to your job.');
        await new Promise(resolve => setTimeout(resolve, 1200));
        continue;
      }
      const job = await response.json();
      if (!response.ok || job.status === 'error') throw new Error(job.error || 'Job failed. Try again.');
      failures = 0;
      onProgress(job);
      if (job.status === 'done') return job.result;
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  }
};
