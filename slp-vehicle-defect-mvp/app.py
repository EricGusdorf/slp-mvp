# --- Sidebar: vehicle selection (changes should NOT affect results until Analyze) ---
with st.sidebar:
    st.header("Vehicle input")

    # Initialize UI state once
    st.session_state.setdefault("ui_input_mode", "Make / Model / Year")
    st.session_state.setdefault("ui_vin", "")
    st.session_state.setdefault("ui_year", 2021)
    st.session_state.setdefault("ui_make", "")
    st.session_state.setdefault("ui_model", "")

    with st.form("vehicle_form", clear_on_submit=False):
        input_mode = st.radio(
            "Lookup by",
            ["VIN", "Make / Model / Year"],
            horizontal=False,
            key="ui_input_mode",
        )

        vin = ""
        make = ""
        model = ""
        year: Optional[int] = None

        if input_mode == "VIN":
            vin = st.text_input(
                "VIN (17 chars)",
                value=st.session_state["ui_vin"],
                placeholder="e.g., 1HGCV1F56MA123456",
                key="ui_vin",
            )
        else:
            year = st.number_input(
                "Model year",
                min_value=1950,
                max_value=datetime.now().year + 1,
                value=int(st.session_state["ui_year"]),
                step=1,
                key="ui_year",
            )

            makes = vp_get_all_makes()
            make = st.selectbox(
                "Make",
                options=[""] + makes,
                index=([""] + makes).index(st.session_state["ui_make"])
                if st.session_state["ui_make"] in ([""] + makes)
                else 0,
                key="ui_make",
            )

            # Only reset UI model when UI make changes (not when year changes)
            prev_ui_make = st.session_state.get("_prev_ui_make", "")
            if make != prev_ui_make:
                st.session_state["ui_model"] = ""
                st.session_state["_prev_ui_make"] = make

            models: list[str] = []
            if make:
                models = vp_get_models_for_make_year(make, int(year))

            # Keep UI model if still valid; otherwise clear
            if st.session_state["ui_model"] and st.session_state["ui_model"] not in models:
                st.session_state["ui_model"] = ""

            options = [""] + models
            model = st.selectbox(
                "Model",
                options=options,
                index=options.index(st.session_state["ui_model"])
                if st.session_state["ui_model"] in options
                else 0,
                key="ui_model",
                disabled=(not make),
            )

        st.divider()
        analyze_clicked = st.form_submit_button("Analyze vehicle", type="primary")
