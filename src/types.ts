export type Test = {
  expr: string;
  want: string;
};

export type Task = {
  id:     string;
  desc:   string;
  tests:  Test[];
};

export type Result = {
  id:     string;
  pass:   boolean;
  bits:   number;
  score:  number;
  errors: string[];
};
