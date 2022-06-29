{
  description = "Flake with Stan";

  # From cmpbayes, 2022-02-21.
  inputs.nixpkgs.url =
    "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";
  inputs.cmpbayes.url = "github:dpaetzel/cmpbayes";
  inputs.cmpbayes.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, cmpbayes }:
    let system = "x86_64-linux";
    in with import nixpkgs {
      inherit system;
    };

    let python = python39;
    in rec {

      devShell.${system} = pkgs.mkShell {

        packages = with python.pkgs; [
          cmpbayes.defaultPackage."${system}"
          ipython
          python
          venvShellHook

          click
          matplotlib
          mlflow
          networkx
          numpy
          pandas
          seaborn
        ];

        # We use ugly venvShellHook here because packaging pystan/httpstan is
        # not entirely straightforward.
        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
        '';
        # Interferes with `nix flake update` and other commands.
        # export LD_LIBRARY_PATH="${
        #   pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
        # }:$LD_LIBRARY_PATH";

        # Using httpstan==4.7.2 (the default as of 2022-06-10) leads to a
        # missing symbols error on NixOS. 4.7.1 works, however, so we use that.
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install httpstan==4.7.1 pystan==3.4.0
        '';

      };
    };
}
