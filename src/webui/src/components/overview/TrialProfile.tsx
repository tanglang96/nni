import * as React from 'react';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../static/const';
import { EXPERIMENT } from '../../static/datamodel';

interface TrialInfoProps {
    experimentUpdateBroadcast: number;
    concurrency: number;
}

class TrialInfo extends React.Component<TrialInfoProps, {}> {

    constructor(props: TrialInfoProps) {
        super(props);
    }

    render() {
        const blacklist = [
            'id', 'logDir', 'startTime', 'endTime',
            'experimentName', 'searchSpace', 'trainingServicePlatform'
        ];
        // tslint:disable-next-line:no-any
        const filter = (key: string, val: any) => {
            if (key === 'trialConcurrency') {
                return this.props.concurrency;
            }
            return blacklist.includes(key) ? undefined : val;
        };
        const profile = JSON.stringify(EXPERIMENT.profile, filter, 2);

        // FIXME: highlight not working?
        return (
            <div className="profile">
                <MonacoEditor
                    width="100%"
                    height="361"
                    language="json"
                    theme="vs-light"
                    value={profile}
                    options={MONACO}
                />
            </div>
        );
    }
}

export default TrialInfo;
