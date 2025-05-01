export interface Assessment {
    name: string;
    url: string;
    description: string;
    job_levels: string[];
    languages: string[];
    duration: number;
    test_types: string[];
    remote_testing: boolean;
    adaptive_irt: boolean;
  }